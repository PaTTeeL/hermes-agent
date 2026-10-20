"""The chain-exhausted 429 wait: which window the retry loop sleeps for, and what the wait says.

When the primary and every fallback are rate-limited there is no backend left to try, so
the turn waits for a rate-limit window instead of failing; each wake re-reads the pools, so
the wait follows the window and may repeat. The wait target follows the
``self_healing_strategy`` setting, the wait is interruptible, and the wait notice names the
target model with its reopen clock and consecutive-429 count.
"""
import logging
import re
import time
from types import SimpleNamespace

import pytest

from agent.conversation_loop import _chain_rate_limit_windows, _do_chain_exhausted_sleep


class _Pool:
    def __init__(self, window_by_model):
        self._windows = {
            model: (value if isinstance(value, tuple) else (value, 1))
            for model, value in window_by_model.items()
        }

    def has_credentials(self):
        return True

    def rate_limit_window_state(self, model_id):
        return self._windows.get(model_id)


def _agent(pool=None, chain=(), model="primary-model"):
    notices = []
    return SimpleNamespace(
        model=model,
        _credential_pool=pool,
        _fallback_chain=list(chain),
        notices=notices,
        _emit_diagnostic_status=notices.append,
        _emit_diagnostic_wait=notices.append,
    )


def _patch_sleep(monkeypatch):
    slept = {}
    monkeypatch.setattr(
        "agent.conversation_loop._interruptible_sleep",
        lambda agent, seconds, status_msg=None: slept.update(seconds=seconds, status_msg=status_msg) or True,
    )
    return slept


def test_waits_for_the_bound_pools_window(monkeypatch):
    """A live window on the bound pool drives the sleep length under the default strategy."""
    slept = _patch_sleep(monkeypatch)
    agent = _agent(pool=_Pool({"primary-model": time.time() + 120}))

    assert _do_chain_exhausted_sleep(agent) is True
    assert slept["seconds"] == pytest.approx(120, abs=5)


def test_no_window_means_no_wait(monkeypatch):
    """No rate-limit record anywhere -> no unlock event; the caller keeps its terminal path."""
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda key: None)
    agent = _agent(pool=_Pool({}))

    assert _do_chain_exhausted_sleep(agent) is False


def test_wait_repeats_without_a_per_call_cap(monkeypatch):
    """Each wake re-reads the pools; a second exhausted event may wait again."""
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


def test_windows_read_each_backends_pool(monkeypatch):
    """Every backend contributes its own earliest per-model window with its ladder count."""
    soon = time.time() + 30
    later = time.time() + 300
    pools = {"fb-a": _Pool({"fb-a-model": (later, 2)}), "fb-b": _Pool({"fb-b-model": (soon, 5)})}
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda key: pools.get(key))
    agent = _agent(
        pool=_Pool({"primary-model": (time.time() + 600, 3)}),
        chain=[{"provider": "fb-a", "model": "fb-a-model"}, {"provider": "fb-b", "model": "fb-b-model"}],
    )

    primary_window, fallback_windows = _chain_rate_limit_windows(agent)

    assert primary_window[0] == "primary-model"
    assert primary_window[1] == pytest.approx(time.time() + 600, abs=5)
    assert primary_window[2] == 3
    assert [w[0] for w in fallback_windows] == ["fb-a-model", "fb-b-model"]
    assert fallback_windows[0][1] == pytest.approx(later, abs=5)
    assert fallback_windows[0][2] == 2
    assert fallback_windows[1][1] == pytest.approx(soon, abs=5)
    assert fallback_windows[1][2] == 5


def test_current_only_waits_in_place(monkeypatch):
    """current_only ignores sooner fallback windows — it waits where it stands."""
    slept = _patch_sleep(monkeypatch)
    monkeypatch.setattr("agent.credential_pool.get_self_healing_strategy", lambda: "current_only")
    pools = {"fb": _Pool({"fb-model": time.time() + 30})}
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda key: pools.get(key))
    agent = _agent(
        pool=_Pool({"primary-model": time.time() + 900}),
        chain=[{"provider": "fb", "model": "fb-model"}],
    )

    assert _do_chain_exhausted_sleep(agent) is True
    assert slept["seconds"] == pytest.approx(900, abs=5)


def test_current_only_without_bound_window_does_not_wait(monkeypatch):
    """current_only with no record on the bound pool -> no wait, no fallback hop."""
    _patch_sleep(monkeypatch)
    monkeypatch.setattr("agent.credential_pool.get_self_healing_strategy", lambda: "current_only")
    pools = {"fb": _Pool({"fb-model": time.time() + 30})}
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda key: pools.get(key))
    agent = _agent(pool=_Pool({}), chain=[{"provider": "fb", "model": "fb-model"}])

    assert _do_chain_exhausted_sleep(agent) is False


def test_strict_sequential_falls_back_to_the_first_chain_window(monkeypatch):
    """strict_sequential: the bound pool first, else the first fallback in chain order."""
    slept = _patch_sleep(monkeypatch)
    monkeypatch.setattr("agent.credential_pool.get_self_healing_strategy", lambda: "strict_sequential")
    pools = {"fb-a": _Pool({"fb-a-model": time.time() + 300}), "fb-b": _Pool({"fb-b-model": time.time() + 30})}
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda key: pools.get(key))
    agent = _agent(
        pool=_Pool({}),
        chain=[{"provider": "fb-a", "model": "fb-a-model"}, {"provider": "fb-b", "model": "fb-b-model"}],
    )

    assert _do_chain_exhausted_sleep(agent) is True
    assert slept["seconds"] == pytest.approx(300, abs=5)


def test_primary_else_fastest_keeps_the_bound_window_when_present(monkeypatch):
    """primary_else_fastest: a bound-pool window is kept even when a fallback reopens sooner."""
    slept = _patch_sleep(monkeypatch)
    monkeypatch.setattr("agent.credential_pool.get_self_healing_strategy", lambda: "primary_else_fastest")
    pools = {"fb": _Pool({"fb-model": time.time() + 30})}
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda key: pools.get(key))
    agent = _agent(
        pool=_Pool({"primary-model": time.time() + 600}),
        chain=[{"provider": "fb", "model": "fb-model"}],
    )

    assert _do_chain_exhausted_sleep(agent) is True
    assert slept["seconds"] == pytest.approx(600, abs=5)


def test_primary_else_fastest_uses_the_soonest_fallback(monkeypatch):
    """Without a bound-pool window, primary_else_fastest takes the soonest fallback."""
    slept = _patch_sleep(monkeypatch)
    monkeypatch.setattr("agent.credential_pool.get_self_healing_strategy", lambda: "primary_else_fastest")
    pools = {"fb-a": _Pool({"fb-a-model": time.time() + 300}), "fb-b": _Pool({"fb-b-model": time.time() + 30})}
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda key: pools.get(key))
    agent = _agent(
        pool=_Pool({}),
        chain=[{"provider": "fb-a", "model": "fb-a-model"}, {"provider": "fb-b", "model": "fb-b-model"}],
    )

    assert _do_chain_exhausted_sleep(agent) is True
    assert slept["seconds"] == pytest.approx(30, abs=5)


def test_fastest_recovery_picks_the_soonest_backend(monkeypatch):
    """fastest_recovery takes the minimum window across the bound pool and fallbacks."""
    slept = _patch_sleep(monkeypatch)
    monkeypatch.setattr("agent.credential_pool.get_self_healing_strategy", lambda: "fastest_recovery")
    pools = {"fb": _Pool({"fb-model": time.time() + 20})}
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda key: pools.get(key))
    agent = _agent(
        pool=_Pool({"primary-model": time.time() + 900}),
        chain=[{"provider": "fb", "model": "fb-model"}],
    )

    assert _do_chain_exhausted_sleep(agent) is True
    assert slept["seconds"] == pytest.approx(20, abs=5)


def test_notice_names_the_target_model_clock_and_count(monkeypatch):
    """The wait notice names the target window: model, local reopen clock, span, 429 streak."""
    _patch_sleep(monkeypatch)
    monkeypatch.setattr("agent.credential_pool.get_self_healing_strategy", lambda: "strict_sequential")
    pools = {"fb-a": _Pool({"fb-a-model": (time.time() + 300, 7)})}
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda key: pools.get(key))
    agent = _agent(pool=_Pool({}), chain=[{"provider": "fb-a", "model": "fb-a-model"}])

    assert _do_chain_exhausted_sleep(agent) is True

    notice = next(msg for msg in agent.notices if "All backends rate-limited" in msg)
    assert "fb-a-model" in notice
    assert "consecutive 429s: 7" in notice
    assert re.search(r"reopens in \d+m\d{2}s", notice)
    assert re.search(r"at \d{2}:\d{2}:\d{2}", notice)


def test_notice_counts_the_primary_ladder_and_ticks_with_its_model(monkeypatch):
    """A bound-pool wait reports the primary's streak; the ticking line names the same model."""
    slept = _patch_sleep(monkeypatch)
    agent = _agent(pool=_Pool({"primary-model": (time.time() + 120, 4)}))

    assert _do_chain_exhausted_sleep(agent) is True

    notice = next(msg for msg in agent.notices if "All backends rate-limited" in msg)
    assert "primary-model" in notice
    assert "consecutive 429s: 4" in notice
    assert "primary-model" in slept["status_msg"]


def test_wait_span_formatting_is_precise():
    """Spans keep seconds under an hour and stay compact past it (never minute-rounded)."""
    from agent.conversation_loop import _fmt_wait_span

    assert _fmt_wait_span(42) == "42s"
    assert _fmt_wait_span(95) == "1m35s"
    assert _fmt_wait_span(300) == "5m00s"
    assert _fmt_wait_span(3725) == "1h02m"


def test_no_wait_reason_is_logged(monkeypatch, caplog):
    """A chain with nothing to wait for says so in the log — the post-hoc trace of a terminal ❌."""
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda key: None)
    agent = _agent(pool=_Pool({}))

    with caplog.at_level(logging.WARNING, logger="agent.conversation_loop"):
        assert _do_chain_exhausted_sleep(agent) is False

    assert "429 chain exhausted — no wait: no rate-limit window on any backend" in caplog.text
