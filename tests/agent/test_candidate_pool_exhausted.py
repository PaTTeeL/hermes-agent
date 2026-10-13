"""Fallback candidate screening: switch targets must be usable right now.

``_candidate_pool_exhausted`` skips a candidate whose pool has nothing usable for the
requested model at this moment — the chain-exhausted wait covers blocked candidates
instead of switching to them (the failure mode behind #89401).
"""
import time
from types import SimpleNamespace
from unittest.mock import patch

from agent.chat_completion_helpers import _candidate_pool_exhausted
from agent.credential_pool import CredentialPool, ModelWindow, PooledCredential


def _entry(id="c1", model_cooldowns=None):
    return PooledCredential(
        provider="fb", id=id, label=id, auth_type="api_key",
        priority=0, source="manual", access_token="k",
        model_cooldowns=model_cooldowns,
    )


def _agent(pool):
    return SimpleNamespace(_credential_pool=pool)


def test_blocked_candidate_is_skipped():
    """An open per-model window leaves nothing usable: not a switch target."""
    now = time.time()
    with patch.object(CredentialPool, "_persist", return_value=None):
        pool = CredentialPool("fb", [
            _entry(model_cooldowns={"m": ModelWindow(reset_at=now + 300, consecutive_count=1)}),
        ])
        assert _candidate_pool_exhausted(_agent(pool), "fb", "m") is True


def test_usable_candidate_is_not_skipped():
    with patch.object(CredentialPool, "_persist", return_value=None):
        pool = CredentialPool("fb", [_entry()])
        assert _candidate_pool_exhausted(_agent(pool), "fb", "m") is False


def test_elapsed_window_is_usable_again():
    """A window that has elapsed no longer blocks the candidate."""
    now = time.time()
    with patch.object(CredentialPool, "_persist", return_value=None):
        pool = CredentialPool("fb", [
            _entry(model_cooldowns={"m": ModelWindow(reset_at=now - 5, consecutive_count=1)}),
        ])
        assert _candidate_pool_exhausted(_agent(pool), "fb", "m") is False


def test_pool_without_credentials_is_not_skipped():
    with patch.object(CredentialPool, "_persist", return_value=None):
        pool = CredentialPool("fb", [])
        assert _candidate_pool_exhausted(_agent(pool), "fb", "m") is False


def test_candidate_pool_load_failure_is_not_skipped(monkeypatch):
    """A load error keeps the allow-through behavior."""
    agent = _agent(SimpleNamespace(provider="other"))
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda key: None)

    assert _candidate_pool_exhausted(agent, "fb", "m") is False
