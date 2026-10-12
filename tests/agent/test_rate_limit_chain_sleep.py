"""The chain-exhausted 429 wait: the retry loop sleeps to the model's window.

When the primary and every fallback are rate-limited there is no backend left to try, so
the turn waits for the current model's earliest window instead of failing; each wake
re-reads the pool, so the wait follows the window and may repeat. The wait is interruptible.
"""
import time
from types import SimpleNamespace

import pytest

from agent.conversation_loop import _do_chain_exhausted_sleep


class _Pool:
    def __init__(self, ttl_by_model):
        self._ttl_by_model = ttl_by_model

    def rate_limit_min_ttl(self, model_id):
        return self._ttl_by_model.get(model_id)


def _agent(pool=None, model="primary-model"):
    return SimpleNamespace(
        model=model,
        _credential_pool=pool,
        _fallback_chain=[],
        _buffer_diagnostic_status=lambda _msg: None,
    )


def test_waits_for_the_current_models_window(monkeypatch):
    """A live window on the current model's pool drives the sleep length."""
    slept = {}
    monkeypatch.setattr(
        "agent.conversation_loop._interruptible_sleep",
        lambda agent, seconds, status_msg=None: slept.update(seconds=seconds) or True,
    )
    agent = _agent(pool=_Pool({"primary-model": time.time() + 120}))

    assert _do_chain_exhausted_sleep(agent) is True
    assert slept["seconds"] == pytest.approx(120, abs=5)


def test_no_window_means_no_wait(monkeypatch):
    """No rate-limit record for the model -> no unlock event; the caller keeps its terminal path."""
    agent = _agent(pool=_Pool({}))

    assert _do_chain_exhausted_sleep(agent) is False


def test_wait_repeats_without_a_per_call_cap(monkeypatch):
    """Each wake re-reads the pool; a second exhausted event may wait again."""
    slept = []
    monkeypatch.setattr(
        "agent.conversation_loop._interruptible_sleep",
        lambda agent, seconds, status_msg=None: slept.append(seconds) or True,
    )
    agent = _agent(pool=_Pool({"primary-model": time.time() + 30}))

    assert _do_chain_exhausted_sleep(agent) is True
    assert _do_chain_exhausted_sleep(agent) is True
    assert len(slept) == 2


def test_interrupted_sleep_reports_false(monkeypatch):
    """An interrupt during the wait surfaces as 'no wait', so the caller stops cleanly."""
    monkeypatch.setattr(
        "agent.conversation_loop._interruptible_sleep",
        lambda agent, seconds, status_msg=None: False,
    )
    agent = _agent(pool=_Pool({"primary-model": time.time() + 30}))

    assert _do_chain_exhausted_sleep(agent) is False
