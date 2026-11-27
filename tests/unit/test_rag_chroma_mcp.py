from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from kama_claude.core.config import McpServerConfig, RagConfig, get_config
from kama_claude.core.mcp.server import McpServerManager
from kama_claude.core.rag.chroma_index import ChromaRagIndex
from kama_claude.core.rag.factory import format_hits


class EmbeddingHandler(BaseHTTPRequestHandler):
    # 接收真实 HTTP 请求并返回可预测向量，避免测试依赖外部 API 或模型下载。
    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        text = body["input"]
        if "embedding_failure" in text:
            self.send_error(503)
            return
        vector = [1.0, 0.0] if "alpha" in text else [0.0, 1.0]
        payload = json.dumps({"data": [{"embedding": vector}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    # 测试 HTTP 服务不向终端输出请求日志。
    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
# 启动本地 embedding 接口，测试结束时关闭线程和端口。
def chroma_config(tmp_path: Path) -> Iterator[RagConfig]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), EmbeddingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield RagConfig(
        backend="chroma",
        chroma_path=str(tmp_path / "chroma"),
        embedding_url=f"http://127.0.0.1:{server.server_port}/embeddings",
        embedding_model="test-vectors",
        chunk_chars=32,
        overlap_chars=0,
    )
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


# 功能：Chroma 检索返回最近来源并清理缩短和删除文件的过期片段。
# 设计：真实持久化库配合正交向量，覆盖重新打开索引与增量同步的边界。
def test_chroma_incremental_and_distance(tmp_path: Path, chroma_config: RagConfig) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    alpha = docs / "alpha.md"
    alpha.write_text("alpha long content\n" * 5, encoding="utf-8")
    beta = docs / "beta.txt"
    beta.write_text("beta unrelated content", encoding="utf-8")
    index = ChromaRagIndex(chroma_config)
    assert index.index_directory(docs) == (2, 0)
    assert index.store.open().count() == 6
    assert index.index_directory(docs) == (0, 2)

    hit = ChromaRagIndex(chroma_config).search("alpha question", 1)[0]
    assert hit.path == str(alpha.resolve())
    assert hit.distance == pytest.approx(0)
    assert "lower is closer" in format_hits([hit])

    alpha.write_text("alpha short", encoding="utf-8")
    assert index.index_directory(docs) == (1, 1)
    assert index.store.open().count() == 2
    beta.unlink()
    index.index_directory(docs)
    assert index.store.open().count() == 1
    alpha.unlink()
    index.index_directory(docs)
    assert index.search("alpha") == []


# 功能：查询未建立或配置不匹配的知识库时明确报错。
# 设计：避免在线服务隐式创建空库，也避免不同模型或切块规则混用同一集合。
def test_chroma_missing_and_signature(tmp_path: Path, chroma_config: RagConfig) -> None:
    with pytest.raises(ValueError, match="not indexed"):
        ChromaRagIndex(chroma_config).search("alpha")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("alpha", encoding="utf-8")
    ChromaRagIndex(chroma_config).index_directory(docs)
    with pytest.raises(ValueError, match="configuration changed"):
        ChromaRagIndex(replace(chroma_config, embedding_model="different")).search("alpha")


# 功能：真实 MCP 子进程能完成发现与查询，并将工具失败包装成错误结果。
# 设计：使用项目自身 Manager 和 Client 对接 SDK stdio，覆盖旧握手和 isError 兼容性。
async def test_stdio_mcp_round_trip(
    tmp_path: Path,
    chroma_config: RagConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="kama_claude.core.mcp.client")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("alpha 中文说明", encoding="utf-8")
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        "[rag]\nbackend = 'chroma'\n"
        f"chroma_path = '{Path(chroma_config.chroma_path).as_posix()}'\n"
        f"embedding_url = '{chroma_config.embedding_url}'\n"
        "embedding_model = 'test-vectors'\nchunk_chars = 32\noverlap_chars = 0\n",
        encoding="utf-8",
    )
    indexed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from kama_claude.cli.main import main; main()",
            "rag",
            "index",
            str(docs),
        ],
        env={**os.environ, "KAMA_CONFIG": str(config_file)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert indexed.returncode == 0, indexed.stderr
    manager = McpServerManager()
    try:
        await manager.start_all(
            [
                McpServerConfig(
                    name="knowledge",
                    command=sys.executable,
                    args=["-m", "kama_claude.core.rag.mcp_server"],
                    env={"KAMA_CONFIG": str(config_file), "PYTHONIOENCODING": "utf-8"},
                )
            ]
        )
        tools = manager.get_tools()
        assert len(tools) == 1
        tool = tools[0]
        assert tool.name == "knowledge__search"
        result = await tool.invoke({"query": "alpha", "top_k": 1})
        assert not result.is_error, result.content
        assert "a.md:1-1" in result.content
        assert "中文说明" in result.content

        invalid = await tool.invoke({"query": "alpha", "top_k": 0})
        assert invalid.is_error
        failed = await tool.invoke({"query": "embedding_failure"})
        assert failed.is_error
    finally:
        await manager.stop_all()


# 功能：新配置能被解析，格式化工具输出不会超过字符预算。
# 设计：真实配置解析排除字段白名单遗漏，较小预算验证材料截断仍保留引用头。
def test_config_and_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from kama_claude.core.rag.index import SearchHit

    config_file = tmp_path / "config.toml"
    config_file.write_text(
        "[rag]\nbackend='chroma'\nchroma_path='data'\n"
        "collection_name='test_docs'\nresult_max_chars=300\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KAMA_CONFIG", str(config_file))
    config = get_config().rag
    assert config.backend == "chroma"
    text = format_hits([SearchHit("a.md", 1, 5, "alpha " * 500, 0.9, 0.1)], 300)
    assert len(text) <= 300
    assert "a.md:1-5" in text
    assert "[truncated]" in text


# 功能：分批写入中断后重新索引会补全片段，不把部分成功的文档当作未变化。
# 设计：让第二批 upsert 失败，验证记录的预期片段总数能识别不完整索引。
def test_interrupted_index_is_repaired(
    tmp_path: Path,
    chroma_config: RagConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("alpha long content\n" * 150, encoding="utf-8")
    index = ChromaRagIndex(chroma_config)
    monkeypatch.setattr(index.embeddings, "embed_texts", lambda texts: [[1.0, 0.0]] * len(texts))
    collection = index.store.open(create=True)
    original = type(collection).upsert
    calls = 0

    # 第二批模拟持久化失败，第一批保留以重现部分成功的状态。
    def fail_second_batch(self: Any, *args: Any, **kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted")
        original(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(type(collection), "upsert", fail_second_batch)
        with pytest.raises(RuntimeError, match="interrupted"):
            index.index_directory(docs)
    assert collection.count() == 128
    assert index.index_directory(docs) == (1, 0)
    assert collection.count() == 150
