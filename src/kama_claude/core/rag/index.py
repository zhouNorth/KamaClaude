from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import httpx

from kama_claude.core.config import RagConfig
from kama_claude.core.rag.chunker import chunk_text


@dataclass(frozen=True)
class SearchHit:
    path: str
    start_line: int
    end_line: int
    text: str
    score: float
    distance: float | None = None


class RagIndex:
    # 初始化知识库，配置决定索引位置及是否使用远程 embedding。
    def __init__(self, config: RagConfig) -> None:
        self.config = config
        self.path = Path(config.index_path).expanduser()

    # 创建索引表，并拒绝将不同 embedding 模型的向量混入同一索引。
    def _open(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA foreign_keys=ON")
        db.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        db.execute(
            "CREATE TABLE IF NOT EXISTS documents (path TEXT PRIMARY KEY, digest TEXT NOT NULL)"
        )
        db.execute("""CREATE TABLE IF NOT EXISTS chunks (
            path TEXT NOT NULL REFERENCES documents(path) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            text TEXT NOT NULL,
            vector TEXT,
            PRIMARY KEY (path, ordinal)
        )""")
        mode = (
            f"{self.config.embedding_url}|{self.config.embedding_model}"
            if self.config.embedding_url
            else "lexical"
        )
        current = db.execute("SELECT value FROM metadata WHERE key='embedding_mode'").fetchone()
        if current and current[0] != mode:
            db.close()
            raise ValueError("RAG embedding configuration changed; rebuild with a new index_path")
        db.execute("INSERT OR IGNORE INTO metadata VALUES ('embedding_mode', ?)", (mode,))
        db.commit()
        return db

    # 将 UTF-8 文本按行切成有重叠的片段，保留可引用的行号。
    def _chunks(self, content: str) -> list[tuple[int, int, str]]:
        return chunk_text(content, self.config.chunk_chars, self.config.overlap_chars)

    # 从兼容 embeddings JSON 响应的 HTTP 服务取得一个文本向量。
    def _embed(self, text: str) -> list[float]:
        key = (
            os.environ.get(self.config.embedding_api_key_env, "")
            if self.config.embedding_api_key_env
            else ""
        )
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        with httpx.Client(timeout=30) as client:
            response = client.post(
                self.config.embedding_url,
                headers=headers,
                json={"model": self.config.embedding_model, "input": text},
            )
            response.raise_for_status()
            vector = response.json()["data"][0]["embedding"]
        if (
            not isinstance(vector, list)
            or not vector
            or not all(isinstance(x, (int, float)) for x in vector)
        ):
            raise ValueError("embedding service returned an invalid vector")
        return [float(x) for x in vector]

    # 增量索引目录内的文本文件，并移除上次索引后已删除的文件。
    def index_directory(self, root: Path) -> tuple[int, int]:
        root = root.resolve()
        if not root.is_dir():
            raise ValueError(f"not a directory: {root}")
        files = sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".md", ".txt", ".rst", ".py"}
            and ".git" not in path.parts
            and ".venv" not in path.parts
        )
        indexed = 0
        skipped = 0
        with closing(self._open()) as db:
            known = {row[0] for row in db.execute("SELECT path FROM documents")}
            seen: set[str] = set()
            for path in files:
                name = str(path.resolve())
                seen.add(name)
                raw = path.read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                old = db.execute("SELECT digest FROM documents WHERE path=?", (name,)).fetchone()
                if old and old[0] == digest:
                    skipped += 1
                    continue
                try:
                    content = raw.decode("utf-8")
                except UnicodeDecodeError:
                    skipped += 1
                    continue
                parts = self._chunks(content)
                vectors = (
                    [self._embed(text) for _, _, text in parts]
                    if self.config.embedding_url
                    else [None] * len(parts)
                )
                with db:
                    db.execute("DELETE FROM documents WHERE path=?", (name,))
                    db.execute("INSERT INTO documents VALUES (?, ?)", (name, digest))
                    db.executemany(
                        "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            (name, i, a, b, text, json.dumps(vector) if vector else None)
                            for i, ((a, b, text), vector) in enumerate(
                                zip(parts, vectors, strict=True)
                            )
                        ),
                    )
                indexed += 1
            with db:
                db.executemany(
                    "DELETE FROM documents WHERE path=?",
                    ((name,) for name in known - seen if Path(name).is_relative_to(root)),
                )
        return indexed, skipped

    # 将中英文文本转换成便于词面检索的词和汉字二元组。
    def _terms(self, text: str) -> Counter[str]:
        lowered = text.lower()
        terms = re.findall(r"[a-z0-9_]+", lowered)
        for group in re.findall(r"[\u3400-\u9fff]+", lowered):
            terms.extend(group[i : i + 2] for i in range(len(group) - 1))
            if len(group) == 1:
                terms.append(group)
        return Counter(terms)

    # 用余弦相似度比较两个 embedding 向量。
    def _cosine(self, left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            raise ValueError("embedding dimensions differ; rebuild the index")
        norm = math.sqrt(sum(x * x for x in left) * sum(x * x for x in right))
        return sum(x * y for x, y in zip(left, right, strict=True)) / norm if norm else 0.0

    # 检索相关片段，返回分数、文本与来源行号。
    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        with closing(self._open()) as db:
            rows = db.execute(
                "SELECT path, start_line, end_line, text, vector FROM chunks"
            ).fetchall()
        if not rows:
            return []
        scores: list[float] = []
        if self.config.embedding_url:
            query_vector = self._embed(query)
            scores = [self._cosine(query_vector, json.loads(row[4])) for row in rows]
        else:
            query_terms = self._terms(query)
            docs = [self._terms(row[3]) for row in rows]
            df = Counter(term for doc in docs for term in doc)
            count = len(rows)
            for doc in docs:
                scores.append(
                    sum(
                        (1 + math.log(1 + doc[term]))
                        * math.log(1 + (count - df[term] + 0.5) / (df[term] + 0.5))
                        for term in query_terms
                        if doc[term]
                    )
                )
        ranked = sorted(zip(rows, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return [
            SearchHit(row[0], row[1], row[2], row[3], score) for row, score in ranked if score > 0
        ][:top_k]
