"""The first error a fallback absorbs must leave a wire dump.

dump_api_request_debug fires only on the terminal path (fallback chain exhausted);
a provider error that a fallback papers over otherwise leaves no wire evidence.
"""

import types

import pytest

from agent import turn_recovery


@pytest.fixture(autouse=True)
def _fresh_dump_keys(monkeypatch):
    """The dedup set is module state: each test starts from an empty one."""
    monkeypatch.setattr(turn_recovery, "_FALLBACK_ABSORBED_DUMP_KEYS", set())


class _StubAgent:
    def __init__(self):
        self.session_id = "s1"
        self.provider = "custom"
        self.model = "m"
        self.log_prefix = ""
        self.dumps = []

    def _dump_api_request_debug(self, api_kwargs, *, reason, error):
        self.dumps.append((reason, error))
        return "/tmp/dump.json"


def _classified(reason="client_error"):
    return types.SimpleNamespace(reason=reason)


def test_dumps_once_per_session_provider_model_reason():
    agent = _StubAgent()
    turn_recovery._dump_request_for_fallback_absorbed_error(
        agent, {"messages": []}, RuntimeError("boom"), _classified())
    turn_recovery._dump_request_for_fallback_absorbed_error(
        agent, {"messages": []}, RuntimeError("boom"), _classified())
    assert len(agent.dumps) == 1
    assert agent.dumps[0][0] == "fallback_absorbed:client_error"


def test_different_reason_dumps_again():
    agent = _StubAgent()
    turn_recovery._dump_request_for_fallback_absorbed_error(
        agent, {}, RuntimeError("x"), _classified("upstream_rate_limit"))
    turn_recovery._dump_request_for_fallback_absorbed_error(
        agent, {}, RuntimeError("x"), _classified("client_error"))
    assert len(agent.dumps) == 2
    assert agent.dumps[0][0] == "fallback_absorbed:upstream_rate_limit"


def test_dump_failure_never_raises():
    agent = _StubAgent()

    def _boom(*args, **kwargs):
        raise RuntimeError("disk full")

    agent._dump_api_request_debug = _boom
    turn_recovery._dump_request_for_fallback_absorbed_error(
        agent, {}, RuntimeError("x"), _classified())
