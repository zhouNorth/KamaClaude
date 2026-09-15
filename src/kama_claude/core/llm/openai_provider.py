from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any

import httpx

from kama_claude.core.bus.events import LlmModelSelectedEvent, LlmTokenEvent, LlmUsageEvent
from kama_claude.core.events.bus import EventBus
from kama_claude.core.llm.types import LlmResponse, ToolCallBlock, UsageStats

_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. "
    "Use the available tools to complete the user's goal. "
    "When the goal is fully achieved, respond with a final answer and do not call any more tools."
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _clean_secret(value: str) -> str:
    return value.strip().strip("<>")


class OpenAIProvider:
    """LLM provider for OpenAI-compatible chat completions APIs."""

    def __init__(
        self,
        model: str,
        *,
        client: httpx.AsyncClient | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        if client is None:
            api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise SystemExit("OPENAI_API_KEY not set")
            base_url = base_url or os.environ.get("OPENAI_BASE_URL")
            if not base_url:
                base_url = os.environ.get("ANTHROPIC_BASE_URL")
            if not base_url:
                raise SystemExit("OPENAI_BASE_URL not set")
            self._client = httpx.AsyncClient()
            self._owns_client = True
        else:
            self._client = client
            self._owns_client = False

        if base_url is None:
            base_url = os.environ.get("OPENAI_BASE_URL")
        if base_url is None:
            base_url = os.environ.get("ANTHROPIC_BASE_URL")
        if base_url is None:
            raise SystemExit("OPENAI_BASE_URL not set")

        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        self._endpoint = base_url + "/chat/completions"
        self._api_key = _clean_secret(api_key or os.environ.get("OPENAI_API_KEY", ""))
        if not self._api_key:
            self._api_key = _clean_secret(os.environ.get("ANTHROPIC_API_KEY", ""))
        if not self._api_key:
            raise SystemExit("OPENAI_API_KEY not set")
        self._model = model

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def chat(
        self,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        bus: EventBus,
        run_id: str,
        *,
        step: int = 0,
    ) -> LlmResponse:
        del step
        await bus.publish(
            LlmModelSelectedEvent(run_id=run_id, model=self._model, strategy="static", ts=_now())
        )

        payload: dict[str, object] = {
            "model": self._model,
            "max_tokens": 4096,
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": self._messages_to_openai(messages),
        }
        tools = self._tools_to_openai(tool_schemas)
        if tools:
            payload["tools"] = tools

        text_parts: list[str] = []
        tool_parts: dict[int, dict[str, str]] = {}
        usage_data: dict[str, Any] = {}
        finish_reason: str | None = None

        async with self._client.stream(
            "POST",
            self._endpoint,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                chunk = json.loads(data)
                if isinstance(chunk.get("usage"), dict):
                    usage_data = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if isinstance(content, str) and content:
                    text_parts.append(content)
                    await bus.publish(
                        LlmTokenEvent(run_id=run_id, token=content, ts=_now())
                    )
                for tool_delta in delta.get("tool_calls") or []:
                    self._accumulate_tool_delta(tool_parts, tool_delta)
                if choice.get("finish_reason"):
                    finish_reason = str(choice["finish_reason"])

        tool_calls = self._build_tool_calls(tool_parts)
        stop_reason = "tool_use" if tool_calls else "end_turn"
        if finish_reason == "tool_calls" and not tool_calls:
            stop_reason = "tool_use"

        input_tokens = int(usage_data.get("prompt_tokens") or 0)
        output_tokens = int(usage_data.get("completion_tokens") or 0)
        prompt_details = usage_data.get("prompt_tokens_details") or {}
        cache_read = int(prompt_details.get("cached_tokens") or 0)
        usage = UsageStats(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
        )
        await bus.publish(
            LlmUsageEvent(
                run_id=run_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_input_tokens=cache_read,
                cache_creation_input_tokens=0,
                ts=_now(),
            )
        )

        return LlmResponse(
            stop_reason=stop_reason,
            tool_calls=tool_calls,
            text="".join(text_parts),
            usage=usage,
        )

    @staticmethod
    def _messages_to_openai(
        messages: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        converted: list[dict[str, object]] = [
            {"role": "system", "content": _SYSTEM_PROMPT}
        ]
        for message in messages:
            role = str(message.get("role", "user"))
            content = message.get("content")
            if role == "assistant" and isinstance(content, list):
                text_parts: list[str] = []
                tool_calls: list[dict[str, object]] = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        text_parts.append(str(block.get("text", "")))
                    elif block.get("type") == "tool_use":
                        tool_calls.append(
                            {
                                "id": str(block.get("id", "")),
                                "type": "function",
                                "function": {
                                    "name": str(block.get("name", "")),
                                    "arguments": json.dumps(
                                        block.get("input", {}), ensure_ascii=False
                                    ),
                                },
                            }
                        )
                assistant: dict[str, object] = {
                    "role": "assistant",
                    "content": "".join(text_parts) or None,
                }
                if tool_calls:
                    assistant["tool_calls"] = tool_calls
                converted.append(assistant)
                continue

            if role == "user" and isinstance(content, list) and all(
                isinstance(block, dict) and block.get("type") == "tool_result"
                for block in content
            ):
                for block in content:
                    assert isinstance(block, dict)
                    result_content = str(block.get("content", ""))
                    if block.get("is_error"):
                        result_content = f"[tool error] {result_content}"
                    converted.append(
                        {
                            "role": "tool",
                            "tool_call_id": str(block.get("tool_use_id", "")),
                            "content": result_content,
                        }
                    )
                continue

            converted.append({"role": role, "content": content})
        return converted

    @staticmethod
    def _tools_to_openai(
        tool_schemas: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": schema.get("name"),
                    "description": schema.get("description", ""),
                    "parameters": schema.get(
                        "input_schema", {"type": "object", "properties": {}}
                    ),
                },
            }
            for schema in tool_schemas
        ]

    @staticmethod
    def _accumulate_tool_delta(
        pending: dict[int, dict[str, str]], tool_delta: dict[str, Any]
    ) -> None:
        index = int(tool_delta.get("index") or 0)
        current = pending.setdefault(
            index, {"id": "", "name": "", "arguments": ""}
        )
        if tool_delta.get("id"):
            current["id"] = str(tool_delta["id"])
        function = tool_delta.get("function") or {}
        if function.get("name"):
            current["name"] = str(function["name"])
        if function.get("arguments"):
            current["arguments"] += str(function["arguments"])

    @staticmethod
    def _build_tool_calls(
        pending: dict[int, dict[str, str]],
    ) -> list[ToolCallBlock]:
        tool_calls: list[ToolCallBlock] = []
        for index in sorted(pending):
            item = pending[index]
            arguments = item["arguments"] or "{}"
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                parsed = {}
            if not isinstance(parsed, dict):
                parsed = {}
            tool_calls.append(
                ToolCallBlock(
                    id=item["id"] or f"call_{index}",
                    name=item["name"],
                    input=parsed,
                )
            )
        return tool_calls
