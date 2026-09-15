from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from kama_claude.core.events.bus import EventBus
from kama_claude.core.llm.openai_provider import OpenAIProvider
from kama_claude.core.llm.types import LlmResponse


def _sse(*chunks: dict[str, Any]) -> str:
    events = [f"data: {json.dumps(chunk)}\n\n" for chunk in chunks]
    events.append("data: [DONE]\n\n")
    return "".join(events)


async def _chat(
    provider: OpenAIProvider,
    messages: list[dict[str, object]],
    tool_schemas: list[dict[str, object]],
) -> tuple[LlmResponse, list[BaseModel]]:
    collected: list[BaseModel] = []
    bus = EventBus()

    async def _collect(event: BaseModel) -> None:
        collected.append(event)

    bus.subscribe(_collect)
    result = await provider.chat(
        messages=messages,
        tool_schemas=tool_schemas,
        bus=bus,
        run_id="r1",
    )
    return result, collected


async def test_openai_streaming_converts_messages_tools_and_events() -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["authorization"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse(
                {
                    "choices": [
                        {"delta": {"content": "Working"}, "finish_reason": None}
                    ]
                },
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "function": {
                                            "name": "read_file",
                                            "arguments": '{"path":',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": '"README.md"}'},
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 5,
                        "prompt_tokens_details": {"cached_tokens": 7},
                    },
                },
            ),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAIProvider(
        "test-model",
        client=client,
        base_url="https://gateway.test",
        api_key="test-key",
    )
    result, events = await _chat(
        provider,
        messages=[
            {"role": "user", "content": "goal"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "calling"},
                    {
                        "type": "tool_use",
                        "id": "call_0",
                        "name": "read_file",
                        "input": {"path": "README.md"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_0",
                        "content": "README content",
                    }
                ],
            },
        ],
        tool_schemas=[
            {
                "name": "read_file",
                "description": "Read a file",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ],
    )
    await client.aclose()

    assert captured["url"] == "https://gateway.test/v1/chat/completions"
    assert captured["authorization"] == "Bearer test-key"
    body = captured["body"]
    assert body["model"] == "test-model"
    assert body["stream"] is True
    assert body["tools"][0]["function"]["parameters"]["required"] == ["path"]
    assert body["messages"][1] == {"role": "user", "content": "goal"}
    assert body["messages"][2]["tool_calls"][0]["function"]["name"] == "read_file"
    assert body["messages"][3] == {
        "role": "tool",
        "tool_call_id": "call_0",
        "content": "README content",
    }

    assert result.stop_reason == "tool_use"
    assert result.text == "Working"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "call_1"
    assert result.tool_calls[0].input == {"path": "README.md"}
    assert result.usage is not None
    assert result.usage.input_tokens == 20
    assert result.usage.output_tokens == 5
    assert result.usage.cache_read_input_tokens == 7

    types = [event.type for event in events]  # type: ignore[attr-defined]
    assert types[0] == "llm.model_selected"
    assert "llm.token" in types
    assert types[-1] == "llm.usage"


async def test_openai_end_turn_has_no_tool_calls() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse(
                {
                    "choices": [
                        {
                            "delta": {"content": "done"},
                            "finish_reason": "stop",
                        }
                    ]
                },
                {
                    "choices": [],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1},
                },
            ),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAIProvider(
        "test-model",
        client=client,
        base_url="https://gateway.test/v1",
        api_key="test-key",
    )
    result, _ = await _chat(provider, messages=[], tool_schemas=[])
    await client.aclose()

    assert result.stop_reason == "end_turn"
    assert result.text == "done"
    assert result.tool_calls == []


def test_openai_missing_key_raises_system_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

    with pytest.raises(SystemExit):
        OpenAIProvider("test-model")
