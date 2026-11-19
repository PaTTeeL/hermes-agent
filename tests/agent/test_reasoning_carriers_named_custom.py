"""Named custom providers must not inherit the local-stack bare ``reasoning`` carrier.

A named custom entry (runtime provider ``custom``) also fronts cloud OpenAI-compatible
endpoints; NVIDIA NIM's parser reads a bare ``reasoning`` key as an alias of
``reasoning_content`` and rejects a message carrying both with HTTP 400
"duplicate field". The carrier table grants the bare key to local stacks only,
through the ``is_local_endpoint`` fallback in ``route_reasoning_carriers``.

The runtime rejected-keys mechanism (``record_reasoning_field_rejection``) is the
safety net for any residual route that still rejects a carrier: a duplicate-field
400 drops the bare key for that (provider, host, model) and the retry rebuilds the
wire without it; a repeat rejection learns nothing, so recovery cannot loop.
"""
import json

from agent.message_sanitization import (
    R,
    RC,
    apply_reasoning_content_policy,
    reasoning_replay_route,
    record_reasoning_field_rejection,
    rejected_reasoning_carriers,
    rejected_reasoning_fields,
    route_reasoning_carriers,
)

NIM_URL = "https://integrate.api.nvidia.com/v1"
NIM_MODEL = "mistralai/mistral-large-4-1t-a52b"

SENT_BOTH = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "ok", "reasoning": "t", "reasoning_content": "t"},
]

# NVIDIA NIM (serde) names the duplicate field; Recursive-style parsers name nothing.
DUPLICATE_BODIES = (
    '{"error":{"message":"Bad request - duplicate field `reasoning_content` in messages[1]"}}',
    '{"error":{"message":"duplicate field `reasoning`"}}',
    '{"error":{"message":"Use one reasoning field per message"}}',
)


def test_named_custom_cloud_route_strips_bare_reasoning():
    carriers = route_reasoning_carriers("custom", NIM_URL)
    assert R not in carriers
    assert RC in carriers


def test_named_custom_colon_form_cloud_route_strips_bare_reasoning():
    # The runtime provider string may keep the ``custom:<name>`` form; the table must
    # not read it as a local stack either.
    carriers = route_reasoning_carriers("custom:nvidia_nim", NIM_URL)
    assert R not in carriers
    assert RC in carriers


def test_named_custom_without_base_url_defaults_to_reasoning_content_only():
    assert route_reasoning_carriers("custom", None) == frozenset({RC})


def test_named_custom_local_stack_keeps_bare_reasoning():
    carriers = route_reasoning_carriers("custom", "http://127.0.0.1:8080/v1")
    assert RC in carriers
    assert R in carriers
    # LAN / unqualified container hosts are local too.
    assert R in route_reasoning_carriers("custom", "http://ollama:11434/v1")


def test_named_custom_wire_carries_no_bare_reasoning_key():
    from agent.transports.chat_completions import ChatCompletionsTransport

    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok", "reasoning": "t", "reasoning_content": "t"},
    ]
    kwargs = ChatCompletionsTransport().build_kwargs(
        "deepseek-ai/deepseek-v4.1-flash", messages,
        provider_name="custom", base_url=NIM_URL,
    )
    raw = json.dumps(kwargs["messages"], ensure_ascii=False)
    assert '"reasoning"' not in raw
    assert '"reasoning_content"' in raw


def test_duplicate_field_bodies_all_learn_the_bare_reasoning_key():
    for body in DUPLICATE_BODIES:
        assert rejected_reasoning_fields(body, SENT_BOTH) == frozenset({R}), body


def test_unknown_field_naming_reasoning_content_takes_all_carriers():
    body = '{"error":{"message":"Extra inputs are not permitted: reasoning_content"}}'
    assert rejected_reasoning_fields(body, SENT_BOTH) == frozenset({R, RC})


class _FakeSessions:
    stored: dict = {}

    @classmethod
    def get_model_config_value(cls, session_id, key, default):
        return cls.stored.get((session_id, key), default)

    @classmethod
    def patch_model_config(cls, session_id, patch):
        for key, value in patch.items():
            cls.stored[(session_id, key)] = value


class _FakePersistence:
    read_only = False
    sessions = _FakeSessions


class _FakeAgent:
    api_mode = "chat_completions"
    provider = "custom"
    base_url = NIM_URL
    model = NIM_MODEL
    session_id = "named-custom-probe"
    persistence = _FakePersistence()
    _session_db = object()

    def __init__(self):
        self.__dict__["_reasoning_rejecting_routes"] = {}


def test_recorded_rejection_drops_bare_key_once_and_never_loops():
    agent = _FakeAgent()
    body = DUPLICATE_BODIES[0]
    new = record_reasoning_field_rejection(agent, body, SENT_BOTH)
    assert new == frozenset({R})
    assert rejected_reasoning_carriers(agent) == frozenset({R})
    # A repeat rejection of the already-recorded key learns nothing: no retry loop.
    assert record_reasoning_field_rejection(agent, body, SENT_BOTH) == frozenset()
    # The rejection is persisted under the (provider, host, model) route key.
    stored = _FakeSessions.stored.get((agent.session_id, "reasoning_rejected_carriers"))
    assert stored and stored[0][:3] == ["custom", "integrate.api.nvidia.com", NIM_MODEL]
    assert stored[0][3] == [R]


def test_rebuilt_wire_after_recorded_rejection_keeps_only_reasoning_content():
    agent = _FakeAgent()
    record_reasoning_field_rejection(agent, DUPLICATE_BODIES[0], SENT_BOTH)
    route = reasoning_replay_route(
        "chat_completions", agent.provider, agent.model, agent.base_url,
        rejected=rejected_reasoning_carriers(agent))
    assert route.carriers == frozenset({RC})

    api_msg = dict(SENT_BOTH[1])
    apply_reasoning_content_policy(SENT_BOTH[1], api_msg, route.pad, route.carriers)
    assert R not in api_msg
    assert api_msg[RC] == "t"

    from agent.transports.chat_completions import ChatCompletionsTransport

    converted = ChatCompletionsTransport().convert_messages(
        SENT_BOTH, provider="custom", base_url=NIM_URL)
    raw = json.dumps(converted, ensure_ascii=False)
    assert '"reasoning"' not in raw
    assert '"reasoning_content"' in raw
