from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

from kama_claude.core.config import RagConfig

if TYPE_CHECKING:
    from chromadb.api import ClientAPI
    from chromadb.api.models.Collection import Collection


class ChromaStore:
    # 保存存储配置和索引签名，延迟打开 Chroma 以缩短 MCP 握手时间。
    def __init__(self, config: RagConfig, embedding_identity: str) -> None:
        self.config = config
        self._client: ClientAPI | None = None
        self._collection: Collection | None = None
        self.signature = hashlib.sha256(
            json.dumps(
                {
                    "schema": 2,
                    "embedding": embedding_identity,
                    "chunk_chars": config.chunk_chars,
                    "overlap_chars": config.overlap_chars,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

    # 创建或打开集合并校验配置，查询路径不自动创建空知识库。
    def open(self, *, create: bool = False) -> Collection:
        if self._collection is not None:
            return self._collection
        import chromadb

        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=str(Path(self.config.chroma_path).expanduser().resolve())
            )
        if create:
            collection = self._client.get_or_create_collection(
                name=self.config.collection_name,
                embedding_function=None,
                metadata={"rag_signature": self.signature},
                configuration={"hnsw": {"space": "cosine"}},
            )
        else:
            try:
                collection = self._client.get_collection(
                    name=self.config.collection_name,
                    embedding_function=None,
                )
            except chromadb.errors.NotFoundError as exc:
                raise ValueError(
                    "knowledge base not indexed; run `kama rag index PATH` first"
                ) from exc
        if (collection.metadata or {}).get("rag_signature") != self.signature:
            raise ValueError("RAG configuration changed; use a new collection_name and reindex")
        self._collection = collection
        return collection
