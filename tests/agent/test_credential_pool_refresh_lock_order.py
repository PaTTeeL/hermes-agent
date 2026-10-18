"""Lock-order and deferral regressions for the single-use refresh path.

Two properties pinned here:

* the ``_auth_store_lock`` refresh section must never wait on the pool lock
  (``self._lock``) — a writer holding the pool lock and waiting for the flock
  used to be stuck behind the old tail's ``_replace_entry``/``_persist`` until
  the flock timeout (the file-lock -> pool-lock inversion). One probe per
  refreshed family: anthropic (``_refresh_anthropic`` + narrow publish),
  openai-codex / xai-oauth (singleton POST + device-code sync), plugin
  (``refresh_credential`` hook);
* a concurrent flush between the store read and its deferred adoption must
  keep the published rotation on disk instead of rolling it back to the
  consumed pair;
* a direct (out-of-lock) call re-reads the singleton before POSTing: a pair
  a peer rotated into auth.json feeds the POST instead of the stale one.
"""
from __future__ import annotations

import json
import threading
import time

import pytest

from agent.credential_pool import CredentialPool, PooledCredential

_NEW_ACCESS = "sk-new-access"
_NEW_REFRESH = "sk-new-refresh"


def _payload(**overrides) -> dict:
    payload = {
        "provider": "anthropic",
        "id": "cred-refresh",
        "label": "refresh-target",
        "auth_type": "oauth",
        "priority": 0,
        "source": "hermes_pkce",
        "access_token": "sk-old-access",
        "refresh_token": "sk-old-refresh",
    }
    payload.update(overrides)
    return payload


def _gated(entered: threading.Event, release: threading.Event, **fields):
    def _refresh(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=10), "refresh step was never released"
        payload = {"access_token": _NEW_ACCESS, "refresh_token": _NEW_REFRESH}
        payload.update(fields)
        return payload
    return _refresh


def _install_leg(monkeypatch, leg: str, entered: threading.Event, release: threading.Event):
    """Arrange one refresh family; returns ``(provider, entry_payload)``."""
    if leg == "anthropic":
        monkeypatch.setattr(
            "agent.anthropic_credentials.refresh_anthropic_oauth_pure",
            _gated(
                entered, release,
                expires_at_ms=int(time.time() * 1000) + 3600_000,
            ),
        )
        return "anthropic", _payload()
    if leg == "openai-codex":
        monkeypatch.setattr(
            "hermes_cli.auth.refresh_codex_oauth_pure", _gated(entered, release),
        )
        return "openai-codex", _payload(
            provider="openai-codex", id="cred-codex", source="device_code",
        )
    if leg == "xai-oauth":
        monkeypatch.setattr(
            "hermes_cli.auth.refresh_xai_oauth_pure", _gated(entered, release),
        )
        return "xai-oauth", _payload(
            provider="xai-oauth", id="cred-xai", source="device_code",
        )
    if leg == "plugin":
        hook = _gated(entered, release)

        def _plugin_refresh_hook(provider):
            return hook if provider == "example-oauth" else None

        monkeypatch.setattr("agent.credential_pool.plugin_refresh_hook", _plugin_refresh_hook)
        return "example-oauth", _payload(
            provider="example-oauth", id="cred-plugin", source="manual",
        )
    raise AssertionError(f"unknown leg {leg}")


@pytest.mark.parametrize("leg", ["anthropic", "openai-codex", "xai-oauth", "plugin"])
def test_locked_refresh_section_does_not_wait_on_the_pool_lock(monkeypatch, tmp_path, leg):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    (tmp_path / "hermes").mkdir()

    post_entered = threading.Event()
    post_release = threading.Event()
    provider, entry_payload = _install_leg(monkeypatch, leg, post_entered, post_release)

    entry = PooledCredential.from_dict(provider, entry_payload)
    pool = CredentialPool(provider, [entry])

    result = {}

    def _refresh():
        result["value"] = pool._refresh_entry(entry, force=True)

    refresh_thread = threading.Thread(target=_refresh)
    refresh_thread.start()
    assert post_entered.wait(timeout=10), "refresh never reached its refresh step"

    # A writer that holds the pool lock and then waits for the auth-store flock.
    writer_done = threading.Event()
    writer_error = {}

    def _writer():
        with pool._lock:
            try:
                pool._persist()
            except Exception as exc:  # pragma: no cover - the regression's failure mode
                writer_error["exc"] = exc
        writer_done.set()

    writer_thread = threading.Thread(target=_writer)
    started = time.monotonic()
    writer_thread.start()
    time.sleep(0.2)  # let the writer park on the flock
    post_release.set()

    assert writer_done.wait(timeout=5), "writer never finished"
    elapsed = time.monotonic() - started
    refresh_thread.join(timeout=5)

    assert not writer_error, f"writer raised: {writer_error.get('exc')!r}"
    assert elapsed < 2.0, (
        f"writer took {elapsed:.2f}s — the locked refresh section waited on the pool lock"
    )
    settled = result.get("value")
    assert settled is not None and settled.access_token == "sk-new-access"
    assert pool.entries()[0].access_token == "sk-new-access"


def test_deferred_adoption_keeps_disk_rotation_against_a_stale_flush(monkeypatch, tmp_path):
    """A concurrent flush between the store read and its deferred adoption must not
    roll the published rotation back to the consumed pair."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    (tmp_path / "hermes").mkdir()

    from hermes_cli.auth import read_credential_pool, write_credential_pool

    old_row = {
        "provider": "anthropic",
        "id": "cred-peer",
        "label": "peer-rotated",
        "auth_type": "oauth",
        "priority": 0,
        "source": "hermes_pkce",
        "access_token": "sk-old-access",
        "refresh_token": "sk-old-refresh",
    }
    entry = PooledCredential.from_dict("anthropic", old_row)
    pool = CredentialPool("anthropic", [entry])
    # load_pool seeds this bookkeeping; the directly-built pool mirrors that seed.
    pool._persisted_token_pairs = {entry.id: ("sk-old-access", "sk-old-refresh")}

    # Another process rotated the pair and published it to the pool store.
    peer_row = dict(old_row, access_token="sk-new-access", refresh_token="sk-new-refresh")
    write_credential_pool("anthropic", [peer_row])

    def _disk_pair():
        row = next(p for p in read_credential_pool("anthropic") if p.get("id") == entry.id)
        return (row.get("access_token"), row.get("refresh_token"))

    # A flush racing ahead of the adoption: stale in-memory row, pre-rotation base.
    pool._persist()
    assert _disk_pair() == ("sk-new-access", "sk-new-refresh"), "stale flush rolled the rotation back"

    original_apply = pool._apply_refresh_settlement

    def flush_then_apply(target, settlement):
        pool._persist()  # third-party flush exactly at the deferral boundary
        return original_apply(target, settlement)

    monkeypatch.setattr(pool, "_apply_refresh_settlement", flush_then_apply)

    result = pool._refresh_entry(entry, force=False)

    assert result is not None and result.access_token == "sk-new-access"
    assert _disk_pair() == ("sk-new-access", "sk-new-refresh")
    assert pool.entries()[0].access_token == "sk-new-access"
    assert pool._persisted_token_pairs[entry.id] == ("sk-new-access", "sk-new-refresh")


def test_direct_call_rereads_peer_rotation_before_posting(monkeypatch, tmp_path):
    """A direct ``_refresh_entry_impl`` call re-reads the singleton first: a pair
    a peer rotated into auth.json feeds the POST instead of the stale one."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()

    entry = PooledCredential.from_dict("xai-oauth", _payload(
        provider="xai-oauth",
        id="cred-xai",
        source="device_code",
        access_token="at-old",
        refresh_token="rt-old",
    ))
    pool = CredentialPool("xai-oauth", [entry])

    # A peer rotated the singleton (auth.json providers.<id>.tokens).
    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {"xai-oauth": {"tokens": {"access_token": "at-peer", "refresh_token": "rt-peer"}}},
    }))

    posted = {}

    def fake_refresh(access_token, refresh_token):
        posted["access_token"], posted["refresh_token"] = access_token, refresh_token
        return {"access_token": "at-final", "refresh_token": "rt-final"}

    monkeypatch.setattr("hermes_cli.auth.refresh_xai_oauth_pure", fake_refresh)

    result = pool._refresh_entry_impl(entry, force=True)

    assert posted == {"access_token": "at-peer", "refresh_token": "rt-peer"}, (
        "the POST must spend the peer-rotated pair re-read from the singleton, "
        f"not the stale one (posted {posted!r})"
    )
    assert result is not None and result.access_token == "at-final"
    assert pool.entries()[0].access_token == "at-final"
    assert pool.entries()[0].refresh_token == "rt-final"
