# KamaClaude

KamaClaude 是一个用 Python 3.12 和 asyncio 实现的终端 AI Agent。你可以让它阅读仓库、修改文件、执行命令，也可以通过 `/orchestrate` 把目标交给规划、执行和审查三个子 Agent。终端实时展示模型输出、工具调用及需要你确认的操作。

[快速开始](#快速开始) · [源码入口](#源码入口) · [阶段分支](#阶段分支) · [源码学习笔记（Word）](https://github.com/zhouNorth/KamaClaude/raw/refs/heads/main/docs/KamaClaude-Study-Notes.docx)

## 快速开始

准备 Python 3.12 和 [uv](https://docs.astral.sh/uv/)。在终端执行：

```bash
git clone https://github.com/zhouNorth/KamaClaude.git
cd KamaClaude
uv sync
cp .env.example .env
```

在仓库根目录的 `.env` 中选择一种模型接口。将下表中的密钥和模型名替换为服务商提供的实际值。

| 接口 | 配置 |
| --- | --- |
| Anthropic | 设置 `ANTHROPIC_API_KEY`。需要指定模型时设置 `KAMA_LLM_DEFAULT_MODEL`。 |
| OpenAI 兼容的 Chat Completions | 设置 `KAMA_LLM_PROVIDER=openai`、`OPENAI_API_KEY`、`OPENAI_BASE_URL` 和 `KAMA_LLM_DEFAULT_MODEL`。基础地址应包含服务商要求的 API 前缀，例如 `/v1`。 |

在第一个终端启动后台服务，保持该终端运行：

```bash
uv run kama-core
```

在第二个终端进入同一仓库，检查连接并打开交互界面：

```bash
uv run kama ping
uv run kama-tui
```

`ping` 返回 `pong` 表示客户端已连上 core。进入 TUI 后可输入「阅读 README.md，说明这个项目怎样执行一次工具调用」。也可以从命令行单独提交任务：

```bash
uv run kama run --goal "总结 README.md 的主要章节"
```

## 一次任务如何执行

1. **接收目标。** CLI 和 TUI 通过本机 TCP 连接 `kama-core`，以换行分隔 JSON-RPC 消息。后台为本次执行分配 `run_id`。
2. **调用模型与工具。** `AgentRunner` 加载历史消息、项目背景和工具定义。`AgentLoop` 根据模型返回的工具名查找实现，执行后把结果送回模型。写文件和 Bash 命令默认需要审批。
3. **展示和保存结果。** 工具进度和模型输出通过事件总线推送到终端。每次运行的事件写入 `events.jsonl`，会话对话写入 `thread.jsonl`。长对话可用 `/compact` 压缩。

## 多 Agent 与知识库

在 TUI 输入以下任务，可以观察父子 Agent 如何交接计划和执行结果：

```text
/orchestrate 新增一个区分大小写的回文判断函数，空字符串返回 True，并补充单元测试。
```

主 Agent 按 Skill 的说明依次调用 planner、executor 和 reviewer。每个子 Agent 有独立的消息上下文和角色工具白名单，父 Agent 将上一阶段的结果写入下一阶段的任务说明。对于相互独立的工作，`spawn_agent` 也支持后台执行，由 `agent_result` 取回结果。

要体验知识库检索，可先在仓库根目录索引源码，再查看命中片段的路径与行号。默认 SQLite 词面检索无需配置 embedding 服务。

```bash
uv run kama rag index ./src
uv run kama rag search "AgentRunner" --top-k 3
```

在 TUI 中使用 `/ask_docs` 可让模型根据索引回答问题。Chroma、向量检索和外接 MCP 搜索工具的配置见 [RAG 指南](RAG.md)。

## 源码入口

| 想了解的机制 | 从这里读起 |
| --- | --- |
| 一次运行如何组装，模型何时继续或停止 | [AgentRunner](src/kama_claude/core/runner.py) · [AgentLoop](src/kama_claude/core/loop.py) |
| 工具如何执行、重试与等待审批 | [工具调用](src/kama_claude/core/tools/invocation.py) · [权限管理](src/kama_claude/core/permissions/manager.py) |
| 会话如何保存，长对话如何压缩 | [会话存储](src/kama_claude/core/session/store.py) · [压缩器](src/kama_claude/core/compact/compactor.py) |
| 主 Agent 如何派生子 Agent 并等待结果 | [编排 Skill](src/kama_claude/core/skills/builtin/orchestrate.md) · [子 Agent 工具](src/kama_claude/core/subagent/tool.py) |
| 终端如何接收响应和实时事件 | [Socket 客户端](src/kama_claude/core/transport/socket_client.py) · [TUI](src/kama_claude/tui/app.py) |

## 阶段分支

`main` 包含完整实现。下列分支保留各阶段的源码，适合按功能的引入顺序阅读。

| 分支 | 阶段内容 |
| --- | --- |
| [s0](https://github.com/zhouNorth/KamaClaude/tree/s0) | 项目骨架、协议与 ping |
| [s1](https://github.com/zhouNorth/KamaClaude/tree/s1) | 模型调用与最小 Agent 循环 |
| [s2](https://github.com/zhouNorth/KamaClaude/tree/s2) | 双进程通信与事件订阅 |
| [s3](https://github.com/zhouNorth/KamaClaude/tree/s3) | 工具调用、任务管理、trace 与 TUI |
| [s4](https://github.com/zhouNorth/KamaClaude/tree/s4) | 多轮会话与持久化 |
| [s5](https://github.com/zhouNorth/KamaClaude/tree/s5) | 工具权限审批与失败处理 |
| [s6](https://github.com/zhouNorth/KamaClaude/tree/s6) | 分层记忆与上下文压缩 |
| [s7](https://github.com/zhouNorth/KamaClaude/tree/s7) | Skill、MCP 与多 Agent 编排 |

[运行手册](RUNBOOK.md) · [IPC 协议](WIRE_PROTOCOL.md) · [测试目录](tests) · [许可证](LICENSE)
