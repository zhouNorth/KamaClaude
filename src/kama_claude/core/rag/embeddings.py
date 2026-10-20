from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import httpx

from kama_claude.core.config import RagConfig

if TYPE_CHECKING:
    from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2


class EmbeddingProvider:
    # 保存统一的文档与查询向量配置，默认使用 Chroma 本地模型。
    def __init__(self, config: RagConfig) -> None:
        self.config = config
        self._local: ONNXMiniLM_L6_V2 | None = None
        self.identity = (
            f"http:{config.embedding_url}:{config.embedding_model}"
            if config.embedding_url
            else "local:chroma-onnx-all-MiniLM-L6-v2"
        )

    # 生成并校验一批文本向量，远程模式兼容单文本 embeddings 接口。
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.config.embedding_url:
            key = os.environ.get(self.config.embedding_api_key_env, "")
            if self.config.embedding_api_key_env and not key:
                raise ValueError(
                    f"missing environment variable: {self.config.embedding_api_key_env}"
                )
            headers = {"Authorization": f"Bearer {key}"} if key else {}
            vectors = []
            with httpx.Client(timeout=25) as client:
                for text in texts:
                    response = client.post(
                        self.config.embedding_url,
                        headers=headers,
                        json={"model": self.config.embedding_model, "input": text},
                    )
                    response.raise_for_status()
                    vectors.append(response.json()["data"][0]["embedding"])
        else:
            from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

            if self._local is None:
                self._local = ONNXMiniLM_L6_V2()
            vectors = self._local(texts)
        converted = [[float(value) for value in vector] for vector in vectors]
        if len(converted) != len(texts):
            raise ValueError("embedding service returned the wrong number of vectors")
        dimensions = {len(vector) for vector in converted}
        if len(dimensions) != 1 or 0 in dimensions:
            raise ValueError("embedding vectors must have the same non-zero dimension")
        if any(not all(math.isfinite(x) for x in v) or not any(v) for v in converted):
            raise ValueError("embedding vectors must be finite and non-zero")
        return converted
