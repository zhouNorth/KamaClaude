from pathlib import Path
from typing import Protocol

from kama_claude.core.config import RagConfig
from kama_claude.core.rag.chroma_index import ChromaRagIndex
from kama_claude.core.rag.index import RagIndex, SearchHit


class RagBackend(Protocol):
    config: RagConfig

    # 返回离线索引的变化文件数和跳过文件数。
    def index_directory(self, root: Path) -> tuple[int, int]: ...

    # 查询知识库并返回带来源的片段。
    def search(self, query: str, top_k: int = 5) -> list[SearchHit]: ...


# 根据配置选择存储后端，原 SQLite 配置仍然可用。
def create_index(config: RagConfig) -> RagBackend:
    return ChromaRagIndex(config) if config.backend == "chroma" else RagIndex(config)


# 限制返回文本总量并保留引用头，距离只作为检索排序信息。
def format_hits(hits: list[SearchHit], max_chars: int = 12000) -> str:
    if not hits:
        return "No passages found in the knowledge base. Do not invent supporting sources."
    parts: list[str] = []
    remaining = max_chars
    for i, hit in enumerate(hits, 1):
        metric = (
            f"cosine_distance={hit.distance:.4f}; lower is closer"
            if hit.distance is not None
            else f"score={hit.score:.3f}"
        )
        header = f"[{i}] {hit.path}:{hit.start_line}-{hit.end_line} ({metric})\n"
        available = remaining - len(header) - 2
        if available <= 20:
            break
        body = hit.text[:available]
        if len(body) < len(hit.text):
            body = body[:-12] + "\n[truncated]"
        parts.append(header + body)
        remaining -= len(header) + len(body) + 2
    return (
        "\n\n".join(parts) or "Source headers exceed the result budget; increase result_max_chars."
    )
