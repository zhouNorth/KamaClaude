# RAG：Chroma + 外接 MCP

离线读取 UTF-8 的 Markdown、TXT、RST、Python 文件，按行切成带重叠的片段并保留来源行号。Chroma 保存片段、向量、文件哈希与元数据。在线 MCP 工具 `knowledge__search` 返回检索材料，最终答案由主 Agent 生成。

## 开始使用

```powershell
uv sync
$env:KAMA_CONFIG = (Resolve-Path examples/rag-mcp.toml).Path
uv run kama rag index ./src
uv run kama rag search "AgentRunner registers tools" --top-k 3
```

先检查 [examples/rag-mcp.toml](examples/rag-mcp.toml) 中 Python 绝对路径和 Chroma 数据目录。示例启用 `backend="chroma"`，未配置远程接口时使用本地 all-MiniLM-L6-v2，第一次索引会下载模型，需要联网，之后可在本地运行。中文资料可配置适合中文的 embedding 服务。本地模型只处理每段输入的前 256 token，较长资料应调小切块大小或使用更长窗口的模型。

在支持现有 daemon 的环境中，两个终端设置相同的 `KAMA_CONFIG`，分别运行 `uv run kama-core` 和 `uv run kama-tui`。TUI 输入：

```text
/ask_knowledge AgentRunner 如何注册工具？
```

守护进程启动 stdio MCP 子进程并发现 `search`，包装为 `knowledge__search`。首次调用可能出现权限审批；允许后检索资料并生成回答。现有 daemon 的平台要求仍适用；MCP 服务和离线命令可单独在 Windows 运行。

`python -m kama_claude.core.rag.mcp_server` 单独运行时会等待 stdin 中的 MCP 报文，通常应由 MCP 客户端启动。

## 配置

可将示例合并进项目 `.kama/config.toml`：

```toml
[rag]
backend = "chroma"
chroma_path = "F:/KamaClaude/KamaClaude/.kama/chroma"
collection_name = "project_docs_v1"
chunk_chars = 1200
overlap_chars = 200
result_max_chars = 12000

[[mcp.servers]]
name = "knowledge"
transport = "stdio"
command = "F:/KamaClaude/KamaClaude/.venv/Scripts/python.exe"
args = ["-m", "kama_claude.core.rag.mcp_server"]
```

`name="knowledge"` 对应 Skill 的 `knowledge__search`。服务继承守护进程环境和工作目录，离线与在线必须使用相同配置。绝对数据路径可以避免从不同目录启动时访问另一份数据库。

## 可选：远程 embedding

在 `[rag]` 中增加：

```toml
embedding_url = "http://127.0.0.1:8000/v1/embeddings"
embedding_model = "your-embedding-model"
embedding_api_key_env = "RAG_EMBEDDING_API_KEY"
```

接口接受 `{ "model": "...", "input": "一段文本" }`，返回 `data[0].embedding` 数字数组。密钥从命名环境变量读取，不放进配置。文档和问题使用同一接口和模型，向量必须维度一致、有限且非零。

更换 embedding 服务、模型或切块规则时，使用新的 `collection_name` 并重建。集合保存配置签名，会拒绝混用不同配置。不要原地替换同名模型的语义后复用旧向量。

## 更新和部署边界

再次索引会跳过未变化的非空文件。修改文件时更新新片段并清理多余旧片段；删除文件时清理该目录范围内的记录，其他目录记录保留。分批更新不是整库原子事务。

本版使用嵌入式 PersistentClient。先完成离线索引再启动在线服务；更新时先停止使用该库的进程，索引后重启。多进程同时持有同一嵌入式库可能造成锁等待。需要多客户端、不停机更新时，应改用独立 Chroma HTTP 服务，或重建新集合后切换。

余弦距离越小越接近，不是置信概率。top-k 可能找到不相关资料，Skill 要求证据不足时说明。材料总字符数受 `result_max_chars` 限制。路径和行号对应最近一次索引版本。

## SQLite 与内置工具

未配置 backend 时保持原 SQLite 模式。默认词面检索无需模型下载；配置 embedding 后使用逐个比较向量的检索。

```toml
[rag]
backend = "sqlite"
index_path = "~/.kama/rag/index.sqlite3"
```

`/ask_docs` 使用内置 `rag_search`，按 backend 选择 SQLite 或 Chroma；`/ask_knowledge` 使用外接 MCP 工具。子 Agent 的工具表目前不自动继承主 Agent 的 MCP 工具。

## 调用链

```text
离线 CLI → create_index(config) → index_directory()
  → chunk_text() → EmbeddingProvider.embed_texts() → Chroma.upsert()

TUI /ask_knowledge → 模型工具调用 → McpTool → McpClient
  → MCP SDK search() → ChromaRagIndex.search()
  → 查询 embedding → Chroma.query() → format_hits()
  → ToolResult → 模型最终回答
```

客户端识别 MCP 的 `isError` 并转换成工具错误。未建库、配置不匹配和 embedding 失败不会伪装成正常的空检索结果。SDK 管理协议 stdout，日志写入 stderr。
