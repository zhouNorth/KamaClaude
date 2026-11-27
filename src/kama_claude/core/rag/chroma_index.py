from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

from kama_claude.core.config import RagConfig
from kama_claude.core.rag.chroma_store import ChromaStore
from kama_claude.core.rag.chunker import chunk_text
from kama_claude.core.rag.embeddings import EmbeddingProvider
from kama_claude.core.rag.index import SearchHit


class ChromaRagIndex:
    # 组装切块、向量生成和 Chroma 存储，复用现有的离线及检索接口。
    def __init__(self, config: RagConfig) -> None:
        # Windows 原生库先在主线程加载，避免首次检索在线程池中导入时卡住。
        importlib.import_module("chromadb")
        if not config.embedding_url:
            importlib.import_module("onnxruntime")
        self.config = config
        self.embeddings = EmbeddingProvider(config)
        self.store = ChromaStore(config, self.embeddings.identity)

    # 增量写入文本片段，并清理缩短或删除文档的过期记录。
    def index_directory(self, root: Path) -> tuple[int, int]:
        import numpy as np

        root = root.resolve()
        if not root.is_dir():
            raise ValueError(f"not a directory: {root}")
        collection = self.store.open(create=True)
        records = collection.get(include=["metadatas"])
        known: dict[str, list[tuple[str, str, int]]] = {}
        for record_id, metadata in zip(records["ids"], records["metadatas"] or [], strict=True):
            if metadata:
                known.setdefault(str(metadata["source"]), []).append(
                    (
                        record_id,
                        str(metadata["content_hash"]),
                        int(str(metadata.get("chunk_count", 0))),
                    )
                )
        files = sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".md", ".txt", ".rst", ".py"}
            and not {".git", ".venv", ".kama", "__pycache__"}.intersection(path.parts)
        )
        changed = skipped = 0
        seen: set[str] = set()
        for path in files:
            source = str(path.resolve())
            seen.add(source)
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            previous = known.get(source, [])
            if (
                previous
                and len(previous) == previous[0][2]
                and all(
                    old_hash == digest and count == len(previous) for _, old_hash, count in previous
                )
            ):
                skipped += 1
                continue
            try:
                content = raw.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise ValueError(f"not a UTF-8 document: {source}") from exc
            chunks = chunk_text(content, self.config.chunk_chars, self.config.overlap_chars)
            vectors = self.embeddings.embed_texts([text for _, _, text in chunks])
            document_id = hashlib.sha256(source.encode()).hexdigest()
            ids = [f"{document_id}:{i}" for i in range(len(chunks))]
            for offset in range(0, len(chunks), 128):
                batch = chunks[offset : offset + 128]
                collection.upsert(
                    ids=ids[offset : offset + 128],
                    documents=[text for _, _, text in batch],
                    embeddings=np.asarray(vectors[offset : offset + 128], dtype=np.float32),
                    metadatas=[
                        {
                            "source": source,
                            "document_id": document_id,
                            "content_hash": digest,
                            "chunk_count": len(chunks),
                            "start_line": start,
                            "end_line": end,
                        }
                        for start, end, _ in batch
                    ],
                )
            current_ids = set(ids)
            obsolete = [record_id for record_id, _, _ in previous if record_id not in current_ids]
            if obsolete:
                collection.delete(ids=obsolete)
            changed += 1
        removed = [
            record_id
            for source, rows in known.items()
            if source not in seen and Path(source).is_relative_to(root)
            for record_id, _, _ in rows
        ]
        if removed:
            collection.delete(ids=removed)
        return changed, skipped

    # 将问题向量交给 Chroma 检索，返回文本、行号和余弦距离。
    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        import numpy as np

        if not query.strip():
            raise ValueError("query must not be empty")
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        collection = self.store.open()
        count = collection.count()
        if not count:
            return []
        result = collection.query(
            query_embeddings=np.asarray(self.embeddings.embed_texts([query]), dtype=np.float32),
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        hits: list[SearchHit] = []
        for document, metadata, distance in zip(
            (result["documents"] or [[]])[0],
            (result["metadatas"] or [[]])[0],
            (result["distances"] or [[]])[0],
            strict=True,
        ):
            if metadata:
                hits.append(
                    SearchHit(
                        path=str(metadata["source"]),
                        start_line=int(str(metadata["start_line"])),
                        end_line=int(str(metadata["end_line"])),
                        text=document or "",
                        score=1.0 - distance,
                        distance=distance,
                    )
                )
        return hits
