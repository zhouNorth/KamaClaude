from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from kama_claude.core.bus.events import LlmTokenEvent
from kama_claude.core.events.bus import EventBus
from kama_claude.core.llm import create_provider


# 功能：验证兼容接口能继续已有会话，并解析流式工具参数和用量。
# 设计：捕获真实 HTTP 请求的模型参数与工具历史，返回分片的 SSE 响应。
async def test_openai_provider_translates_history_and_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/compatible-mode/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "local-test")
    chunks = [
        {"choices": [{"delta": {"content": "Checking"}}]},
        {"choices": [{"delta": {"tool_calls": [{
            "index": 0, "id": "call-new", "function": {"name": "check", "arguments": '{"x":'},
        }]}}]},
        {"choices": [{"delta": {"tool_calls": [{
            "index": 0, "function": {"arguments": "1}"},
        }]}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 3}},
    ]

    # 检查消息顺序和工具定义，再返回模拟模型响应
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/compatible-mode/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer local-test"
        payload = json.loads(request.content)
        assert payload["model"] == "qwen-flash"
        assert [message["role"] for message in payload["messages"]] == [
            "system", "user", "assistant", "tool",
        ]
        assert payload["messages"][2]["tool_calls"][0]["id"] == "call-old"
        assert payload["messages"][3]["tool_call_id"] == "call-old"
        assert payload["tools"][0]["function"]["parameters"] == {"type": "object"}

        events = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        return httpx.Response(200, text=events + "data: [DONE]\n\n")

    client_type = httpx.AsyncClient

    # 使用模拟传输拦截请求，无需访问外网
    def mock_client(**kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(respond)
        return client_type(**kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", mock_client)
    bus = EventBus()
    tokens: list[str] = []

    # 记录供终端界面消费的文本事件
    async def collect(event: BaseModel) -> None:
        if isinstance(event, LlmTokenEvent):
            tokens.append(event.token)

    bus.subscribe(collect)
    result = await create_provider("openai", "qwen-flash").chat(
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "call-old", "name": "check", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call-old", "content": "OK"},
            ]},
        ],
        [{"name": "check", "input_schema": {"type": "object"}}],
        bus,
        "run-test",
        system="Please check.",
    )

    assert result.stop_reason == "tool_use"
    assert result.text == "Checking"
    assert tokens == ["Checking"]
    assert result.tool_calls[0].id == "call-new"
    assert result.tool_calls[0].input == {"x": 1}
    assert result.usage and result.usage.output_tokens == 3
