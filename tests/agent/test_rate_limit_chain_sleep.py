"""The chain-exhausted 429 wait: which window the retry loop sleeps for.

When the primary and every fallback are rate-limited there is no backend left to try, so
the turn waits once for the earliest window instead of failing. The wait target comes from
each backend's credential pool (``rate_limit_min_ttl``), and the three strategies pick
between the primary and the fallback chain.
"""
import time
from types import SimpleNamespace

import pytest

from agent.conversation_loop import _chain_rate_limit_ttls, _do_chain_exhausted_sleep


class _Pool:
    def __init__(self, ttl_by_model):
        self._ttl_by_model = ttl_by_model

    def has_credentials(self):
        return True

    def rate_limit_min_ttl(self, model_id):
        return self._ttl_by_model.get(model_id)


def _agent(primary_pool=None, chain=(), model="primary-model"):
    return SimpleNamespace(
        model=model,
        _credential_pool=primary_pool,
        _fallback_chain=list(chain),
        _buffer_diagnostic_status=lambda _msg: None,
    )


def test_ttls_read_the_primary_pool_and_each_fallback(monkeypatch):
    """Every backend contributes its own earliest per-model window."""
    soon = time.time() + 30
    later = time.time() + 300
    pools = {"fb-a": _Pool({"fb-a-model": later}), "fb-b": _Pool({"fb-b-model": soon})}
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda key: pools.get(key))
    agent = _agent(
        primary_pool=_Pool({"primary-model": time.time() + 600}),
        chain=[{"provider": "fb-a", "model": "fb-a-model"}, {"provider": "fb-b", "model": "fb-b-model"}],
    )

    primary_ttl, first_ttl, min_ttl = _chain_rate_limit_ttls(agent)

    assert primary_ttl == pytest.approx(time.time() + 600, abs=5)
    assert first_ttl == pytest.approx(later, abs=5)
    assert min_ttl == pytest.approx(soon, abs=5)


def test_no_rate_limit_anywhere_means_no_wait(monkeypatch):
    """A non-rate-limit failure has no unlock event, so the caller keeps its terminal path."""
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda key: None)
    agent = _agent(primary_pool=_Pool({}), chain=[])

    assert _do_chain_exhausted_sleep(agent) is False


def test_exhausted_chain_waits_for_the_primary_window(monkeypatch):
    """With the primary rate-limited, strict_sequential waits for the primary's own reset."""
    slept = {}
    monkeypatch.setattr("agent.credential_pool.get_fallback_strategy", lambda: "strict_sequential")
    monkeypatch.setattr(
        "agent.conversation_loop._interruptible_sleep",
        lambda agent, seconds, status_msg=None: slept.update(seconds=seconds) or True,
    )
    agent = _agent(primary_pool=_Pool({"primary-model": time.time() + 120}), chain=[])

    assert _do_chain_exhausted_sleep(agent) is True
    assert slept["seconds"] == pytest.approx(120, abs=5)


def test_fastest_recovery_picks_the_soonest_backend(monkeypatch):
    """fastest_recovery takes the minimum window across primary and fallbacks."""
    slept = {}
    monkeypatch.setattr("agent.credential_pool.get_fallback_strategy", lambda: "fastest_recovery")
    monkeypatch.setattr(
        "agent.conversation_loop._interruptible_sleep",
        lambda agent, seconds, status_msg=None: slept.update(seconds=seconds) or True,
    )
    pools = {"fb": _Pool({"fb-model": time.time() + 20})}
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda key: pools.get(key))
    agent = _agent(
        primary_pool=_Pool({"primary-model": time.time() + 900}),
        chain=[{"provider": "fb", "model": "fb-model"}],
    )

    assert _do_chain_exhausted_sleep(agent) is True
    assert slept["seconds"] == pytest.approx(20, abs=5)
