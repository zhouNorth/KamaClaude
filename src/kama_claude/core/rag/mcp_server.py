from __future__ import annotations

import asyncio
import logging

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from kama_claude.core.config import RagConfig, get_config
from kama_claude.core.rag.factory import create_index, format_hits

logger = logging.getLogger(__name__)


# 注册检索工具并保持知识库句柄，启动过程不下载 embedding 模型。
def build_server(config: RagConfig) -> FastMCP:
    index = create_index(config)
    server = FastMCP("KamaKnowledge")

    @server.tool(
        name="search",
        description="Search project knowledge; return source passages and line numbers.",
    )
    # 检索已有索引并返回文本；异常使用 MCP 工具错误明确回传。
    async def search(query: str, top_k: int = 5) -> str:
        try:
            logger.info("Knowledge search started")
            hits = await asyncio.to_thread(index.search, query, top_k)
            logger.info("Knowledge search completed with %d passages", len(hits))
            return format_hits(hits, config.result_max_chars)
        except Exception as exc:
            raise ToolError(f"Knowledge search failed: {exc}") from exc

    return server


# 使用继承的配置启动 stdio 服务，协议输出由 SDK 管理。
def main() -> None:
    build_server(get_config().rag).run(transport="stdio")


if __name__ == "__main__":
    main()
