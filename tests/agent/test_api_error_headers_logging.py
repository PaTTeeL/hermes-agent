"""API error dumps carry the response headers, and 429 retry lines name the bucket.

A provider's rate-limit headers (``x-ratelimit-*``, ``Retry-After``) are what tell
"wait out the window" apart from "switch provider", and they only exist on the error
response. The dump keeps the full header set for post-hoc analysis; the retry warning
line carries the bucket fields inline because it is the only per-attempt record.

Regression: adding the headers must not put a credential on disk — a JSON-serialized
header map is invisible to the plain ``name: value`` redaction shapes.
"""
import json
import logging
from types import SimpleNamespace

import httpx
import pytest

import run_agent
from agent.agent_runtime_helpers import _api_error_debug_info
from agent.turn_recovery import _rate_limit_note_from_error, log_api_error_attempt

NIM_HEADERS = {
    "x-ratelimit-remaining-requests": "0",
    "x-ratelimit-limit-requests": "60",
    "x-ratelimit-reset-requests": "8s",
    "retry-after": "8",
    "x-request-id": "req_abc123",
}


def _patch_agent_bootstrap(monkeypatch):
    monkeypatch.setattr(
        "model_tools.get_tool_definitions",
        lambda **kwargs: [
            {
                "type": "function",
                "function": {
                    "name": "terminal",
                    "description": "Run shell commands.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda: {})


def _build_agent(monkeypatch):
    _patch_agent_bootstrap(monkeypatch)
    agent = run_agent.AIAgent(
        model="z-ai/glm-5.3",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="nvapi-test-key",
        quiet_mode=True,
        max_iterations=4,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._cleanup_task_resources = lambda task_id: None
    agent._persist_session = lambda messages, history=None: None
    return agent


def _rate_limit_error(headers: dict) -> Exception:
    response = httpx.Response(
        status_code=429,
        headers=headers,
        text='{"error": {"message": "rate limited"}}',
        request=httpx.Request("POST", "https://integrate.api.nvidia.com/v1/chat/completions"),
    )

    class _FakeAPIError(Exception):
        pass

    error = _FakeAPIError("rate limited")
    error.status_code = 429
    error.response = response
    error.body = {"error": {"message": "rate limited"}}
    return error


def test_api_error_debug_info_carries_response_headers():
    """The error detail dict every request dump embeds now names the response headers."""
    info = _api_error_debug_info(_rate_limit_error(NIM_HEADERS))

    assert info["response_headers"]["x-ratelimit-remaining-requests"] == "0"
    assert info["response_headers"]["retry-after"] == "8"
    assert info["response_headers"]["x-request-id"] == "req_abc123"


def test_api_error_debug_info_without_response_has_no_headers():
    """A connection error carries no response object; the key stays absent, not null."""
    info = _api_error_debug_info(SimpleNamespace(status_code=None))

    assert "response_headers" not in info


def test_dump_writes_response_headers_without_leaking_credentials(monkeypatch, tmp_path):
    """The dump keeps the rate-limit headers and masks an echoed credential and cookie."""
    agent = _build_agent(monkeypatch)
    agent.logs_dir = tmp_path
    error = _rate_limit_error({
        **NIM_HEADERS,
        "authorization": "Bearer nvapi-REALSECRETVALUE9876543210",
        "set-cookie": "session=REALSESSIONCOOKIE9876543210",
    })

    dump_file = agent._dump_api_request_debug(
        {"model": "z-ai/glm-5.3", "messages": []}, reason="non_retryable_client_error", error=error,
    )

    payload = json.loads(dump_file.read_text(encoding="utf-8"))
    headers = payload["error"]["response_headers"]
    assert headers["x-ratelimit-remaining-requests"] == "0"
    assert headers["retry-after"] == "8"
    dumped = json.dumps(payload)
    assert "REALSECRETVALUE9876543210" not in dumped
    assert "REALSESSIONCOOKIE9876543210" not in dumped


def test_rate_limit_note_names_the_bucket():
    """Only a response that carries rate-limit headers produces a note."""
    note = _rate_limit_note_from_error(_rate_limit_error(NIM_HEADERS))

    assert "x-ratelimit-remaining-requests=0" in note
    assert "retry-after=8s" in note


def test_rate_limit_note_is_empty_without_rate_limit_headers():
    """A non-429 error (or a 429 without bucket headers) adds no note to the line."""
    error = _rate_limit_error({"x-request-id": "req_abc123"})

    assert _rate_limit_note_from_error(error) == ""


def test_warning_line_carries_the_rate_limit_note(monkeypatch, caplog):
    """The always-logged retry line names the bucket so the wait is visible in agent.log."""
    agent = _build_agent(monkeypatch)
    agent.verbose_logging = False

    with caplog.at_level(logging.WARNING, logger="agent.turn_recovery"):
        log_api_error_attempt(
            agent, _rate_limit_error(NIM_HEADERS), retry_count=1, max_retries=3,
            status_code=429, elapsed_time=1.5, api_messages=[{"role": "user", "content": "hi"}],
            approx_tokens=10,
        )

    line = next(r.getMessage() for r in caplog.records if "API call failed" in r.getMessage())
    assert "x-ratelimit-remaining-requests=0" in line
    assert "retry-after=8s" in line
