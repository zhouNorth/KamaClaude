from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any

import httpx

from kama_claude.core.bus.events import LlmModelSelectedEvent, LlmTokenEvent, LlmUsageEvent
from kama_claude.core.events.bus import EventBus
from kama_claude.core.llm.types import LlmResponse, ToolCallBlock, UsageStats

_STOP_REASONS = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens"}


# 将会话里存储的内容块转换为 Chat Completions 消息和工具调用
def _convert_messages(
    messages: list[dict[str, object]], system: str | None
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    if system:
        converted.append({"role": "system", "content": system})
    for message in messages:
        role, content = message["role"], message["content"]
        if isinstance(content, str):
            converted.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            raise ValueError("Unsupported conversation message content")

        text: list[str] = []
        calls: list[dict[str, Any]] = []
        for block in content:
            kind = block["type"]
            if kind == "text":
                text.append(block["text"])
            elif kind == "tool_use":
                calls.append({
                    "id": block["id"],
                    "type": "function",
                    "function": {"name": block["name"], "arguments": json.dumps(block["input"])},
                })
            elif kind == "tool_result":
                converted.append({
                    "role": "tool", "tool_call_id": block["tool_use_id"],
                    "content": block["content"],
                })
        if calls:
            converted.append({
                "role": role, "content": "\n".join(text) or None, "tool_calls": calls,
            })
        elif text:
            converted.append({"role": role, "content": "\n".join(text)})
    return converted


class OpenAICompatibleProvider:
    # 从环境变量读取聊天接口配置
    def __init__(self, model: str) -> None:
        self._model = model
        self._api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        self._base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
        self._direct = os.environ.get("KAMA_OPENAI_DIRECT") == "1"
        if not self._api_key or not self._base_url:
            raise SystemExit("OPENAI_API_KEY and OPENAI_BASE_URL must be set")

    # 读取流式文本、工具调用和 token 用量并转换成内部响应
    async def chat(
        self,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        bus: EventBus,
        run_id: str,
        *,
        step: int = 0,
        system: str | None = None,
    ) -> LlmResponse:
        await bus.publish(LlmModelSelectedEvent(
            run_id=run_id, model=self._model, strategy="static", ts=datetime.now(UTC).isoformat(),
        ))
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": _convert_messages(messages, system),
            "max_tokens": 8192,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tool_schemas:
            payload["tools"] = [
                {"type": "function", "function": {
                    "name": tool["name"], "description": tool.get("description", ""),
                    "parameters": tool["input_schema"],
                }}
                for tool in tool_schemas
            ]

        text: list[str] = []
        calls: dict[int, dict[str, str]] = {}
        finish_reason = ""
        usage: dict[str, Any] = {}
        # 显式传输层绕过环境代理，同时保留默认的 TLS 证书配置
        transport = httpx.AsyncHTTPTransport() if self._direct else None
        async with httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=httpx.Timeout(120, connect=15),
            transport=transport,
        ) as client, client.stream("POST", "chat/completions", json=payload) as response:
            if response.status_code == 403:
                await response.aread()
                try:
                    error_code = response.json().get("error", {}).get("code")
                except ValueError:
                    error_code = None
                if error_code == "AccessDenied.Unpurchased":
                    raise PermissionError("Model Studio returned HTTP 403 AccessDenied.Unpurchased")
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("error"):
                    raise RuntimeError("Chat Completions returned an error in the stream")
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices", []):
                    finish_reason = choice.get("finish_reason") or finish_reason
                    delta = choice.get("delta") or {}
                    token = delta.get("content")
                    if token:
                        text.append(token)
                        await bus.publish(LlmTokenEvent(
                            run_id=run_id, token=token, ts=datetime.now(UTC).isoformat(),
                        ))
                    for part in delta.get("tool_calls") or []:
                        call = calls.setdefault(part["index"], {"id": "", "name": "", "args": ""})
                        function = part.get("function") or {}
                        call["id"] += part.get("id") or ""
                        call["name"] += function.get("name") or ""
                        call["args"] += function.get("arguments") or ""
        stop_reason = _STOP_REASONS.get(finish_reason)
        if stop_reason is None:
            raise RuntimeError(
                f"Chat Completions stream ended with finish_reason={finish_reason!r}"
            )

        prompt_details = usage.get("prompt_tokens_details") or {}
        stats = UsageStats(
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            cache_read_input_tokens=prompt_details.get("cached_tokens", 0),
        )
        if usage:
            await bus.publish(LlmUsageEvent(
                run_id=run_id, input_tokens=stats.input_tokens, output_tokens=stats.output_tokens,
                cache_read_input_tokens=stats.cache_read_input_tokens,
                cache_creation_input_tokens=stats.cache_creation_input_tokens,
                ts=datetime.now(UTC).isoformat(),
            ))
        return LlmResponse(
            stop_reason=stop_reason,
            text="".join(text),
            usage=stats if usage else None,
            tool_calls=[
                ToolCallBlock(
                    id=call["id"], name=call["name"],
                    input={} if stop_reason == "max_tokens" else json.loads(call["args"] or "{}"),
                )
                for _, call in sorted(calls.items())
            ],
        )
