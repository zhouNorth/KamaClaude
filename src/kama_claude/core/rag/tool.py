from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, Field

from kama_claude.core.rag.factory import RagBackend, format_hits
from kama_claude.core.tools.base import BaseTool, ToolResult


class RagSearchParams(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


class RagSearchTool(BaseTool):
    name = "rag_search"
    description = (
        "Search indexed project knowledge and return quoted source passages with line numbers."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Question or search terms"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
        },
        "required": ["query"],
    }
    params_model = RagSearchParams

    # 保存索引引用，供每次工具调用时检索。
    def __init__(self, index: RagBackend) -> None:
        self._index = index

    # 在线程中执行 SQLite 检索，并将来源片段返回给 Agent。
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        p = RagSearchParams.model_validate(params)
        try:
            hits = await asyncio.to_thread(self._index.search, p.query, p.top_k)
        except Exception as exc:
            return ToolResult(
                content=f"RAG search failed: {exc}", is_error=True, error_type="runtime_error"
            )
        return ToolResult(content=format_hits(hits, self._index.config.result_max_chars))
