import os
import subprocess
import sys
from pathlib import Path

import pytest

from kama_claude.core.config import RagConfig
from kama_claude.core.rag.index import RagIndex
from kama_claude.core.rag.tool import RagSearchTool


# 功能：索引后能检索来源行号，文件改动和删除会更新索引。
# 设计：使用临时目录和真实 SQLite 库覆盖离线增量流程与在线查询的连接处。
def test_index_search_update_and_delete(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    file = docs / "guide.md"
    file.write_text(
        "# AgentRunner\n\nAgentRunner registers tools for AgentLoop.\n", encoding="utf-8"
    )
    index = RagIndex(RagConfig(index_path=str(tmp_path / "index.sqlite3")))

    assert index.index_directory(docs) == (1, 0)
    hits = index.search("registers tools")
    assert len(hits) == 1
    assert hits[0].path == str(file.resolve())
    assert hits[0].start_line == 1
    assert index.index_directory(docs) == (0, 1)

    file.write_text("# AgentRunner\n\nAgentRunner stores notes.\n", encoding="utf-8")
    assert index.index_directory(docs) == (1, 0)
    assert index.search("registers tools") == []
    file.unlink()
    index.index_directory(docs)
    assert index.search("stores notes") == []


# 功能：RAG 工具将索引命中包装成带来源的 ToolResult。
# 设计：经过真实 invoke 入口验证异步工具和同步 SQLite 检索能正确衔接。
async def test_rag_tool_returns_source(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.txt").write_text("The context stores the goal.\n", encoding="utf-8")
    index = RagIndex(RagConfig(index_path=str(tmp_path / "index.sqlite3")))
    index.index_directory(docs)

    result = await RagSearchTool(index).invoke({"query": "context goal"})
    assert not result.is_error
    assert "notes.txt:1-1" in result.content
    assert "context stores the goal" in result.content


# 功能：启用 embedding 时索引和查询都使用向量相似度，并拒绝混用索引模式。
# 设计：替换远程 embedding 调用，固定两个正交向量以验证排序和模式边界。
def test_embedding_search_and_mode_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "alpha.txt").write_text("alpha topic", encoding="utf-8")
    (docs / "beta.txt").write_text("beta topic", encoding="utf-8")
    path = str(tmp_path / "index.sqlite3")
    index = RagIndex(
        RagConfig(
            index_path=path,
            embedding_url="http://localhost/embeddings",
            embedding_model="example",
        )
    )
    monkeypatch.setattr(index, "_embed", lambda text: [1.0, 0.0] if "alpha" in text else [0.0, 1.0])
    index.index_directory(docs)
    hits = index.search("alpha question")
    assert [Path(hit.path).name for hit in hits] == ["alpha.txt"]
    with pytest.raises(ValueError, match="embedding configuration changed"):
        RagIndex(RagConfig(index_path=path)).search("alpha")


# 功能：用户可通过 kama rag index 和 search 完成离线与在线检索流程。
# 设计：在独立进程使用临时配置调用真实 CLI 入口，避免污染用户的默认索引。
def test_rag_cli_round_trip(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("AgentRunner registers tools.\n", encoding="utf-8")
    config = tmp_path / "config.toml"
    config.write_text(
        f'[rag]\nindex_path = "{(tmp_path / "index.sqlite3").as_posix()}"\n',
        encoding="utf-8",
    )
    env = {**os.environ, "KAMA_CONFIG": str(config)}
    entry = [sys.executable, "-c", "from kama_claude.cli.main import main; main()", "rag"]
    indexed = subprocess.run(
        [*entry, "index", str(docs)], env=env, capture_output=True, text=True, check=False
    )
    searched = subprocess.run(
        [*entry, "search", "registers tools"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert indexed.returncode == 0, indexed.stderr
    assert searched.returncode == 0, searched.stderr
    assert "guide.md:1-1" in searched.stdout
