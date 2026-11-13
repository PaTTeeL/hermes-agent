"""Named custom providers must not inherit the local-stack bare ``reasoning`` carrier.

A named custom entry (runtime provider ``custom``) also fronts cloud OpenAI-compatible
endpoints; NVIDIA NIM's parser reads a bare ``reasoning`` key as an alias of
``reasoning_content`` and rejects a message carrying both with HTTP 400
"duplicate field". The carrier table grants the bare key to local stacks only,
through the ``is_local_endpoint`` fallback in ``route_reasoning_carriers``.
"""

import json

from agent.message_sanitization import R, RC, route_reasoning_carriers

NIM_URL = "https://integrate.api.nvidia.com/v1"


def test_named_custom_cloud_route_strips_bare_reasoning():
    carriers = route_reasoning_carriers("custom", NIM_URL)
    assert R not in carriers
    assert RC in carriers


def test_named_custom_local_stack_keeps_bare_reasoning():
    carriers = route_reasoning_carriers("custom", "http://127.0.0.1:8080/v1")
    assert RC in carriers
    assert R in carriers


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
