# KamaClaude

KamaClaude 是一个用 Python 3.12 和 asyncio 实现的本地 AI Agent 项目。用户在终端输入目标，后台进程调用模型、执行工具，并把模型输出、工具进度和权限请求推送回终端。

**[下载《KamaClaude 项目源码学习笔记》（Word，17 页）](./docs/KamaClaude-Study-Notes.docx)**

笔记从一次 `kama ping` 开始，沿着实际代码追踪请求和响应，再用“总结 README”和“编写回文函数”两个案例讲解单 Agent 执行与多 Agent 协作。文中保留了函数名、参数说明和源码路径。main 分支包含完整源码、测试和学习笔记。

## 项目如何运行

- **终端与后台通信**　`kama` 和 `kama-tui` 通过本机 TCP 连接 `kama-core`。每条 JSON-RPC 消息独占一行。客户端按请求 ID 匹配响应，并通过事件订阅接收运行进度。
- **工具调用与人工审批**　模型根据工具描述选择操作，`AgentLoop` 将工具结果写回上下文。写文件、运行 Bash 等操作按策略放行、拒绝或等待用户审批。
- **会话与上下文**　会话保存对话历史和笔记，每次运行单独记录事件。长对话可以生成摘要，用户也可以通过 `/compact` 压缩已保存的会话历史。
- **多 Agent 协作**　`/orchestrate` 引导主 Agent 依次委派规划、执行和审查。子 Agent 各自维护消息上下文和工具白名单，主 Agent 将前一阶段的结果传给下一阶段。
- **知识检索与外部工具**　RAG 工具从已索引的文档中返回带来源的片段。MCP 客户端发现外部服务提供的工具，将其注册到 Agent 使用的工具表中。

## 按问题阅读

| 想了解的问题 | 笔记页码 |
| --- | --- |
| 一次请求如何穿过 TCP？`Future`、`ContextVar` 和事件循环各做什么？ | 2–5 |
| 模型如何调用工具？如何区分任务状态、执行事件和 trace？ | 6–8 |
| 守护进程崩溃后留下哪些数据？权限审批如何唤醒工具调用？ | 9–10 |
| 全局背景、会话笔记和消息历史有什么区别？两种压缩何时生效？ | 11–12 |
| Skill 与 MCP 如何接入？主 Agent 怎样等待和汇总子 Agent 的结果？ | 13–15 |
| 文档如何建立索引并返回出处？怎样从运行现象定位到源码？ | 16–17 |

## 运行项目

使用 Python 3.12 和 uv。在仓库根目录执行 `uv sync`，复制 `.env.example` 为 `.env` 并配置模型凭据。启动 `uv run kama-core` 后，打开另一个终端，先运行 `uv run kama ping` 确认连接，再运行 `uv run kama-tui`。

## 阶段分支

| 分支 | 阅读重点 |
| --- | --- |
| [s0](https://github.com/zhouNorth/KamaClaude/tree/s0) | 项目骨架、协议与 ping |
| [s1](https://github.com/zhouNorth/KamaClaude/tree/s1) | 模型调用与最小 Agent 循环 |
| [s2](https://github.com/zhouNorth/KamaClaude/tree/s2) | 客户端与守护进程通信 |
| [s3](https://github.com/zhouNorth/KamaClaude/tree/s3) | 工具调用、任务管理与 TUI |
| [s4](https://github.com/zhouNorth/KamaClaude/tree/s4) | 多轮会话与持久化 |
| [s5](https://github.com/zhouNorth/KamaClaude/tree/s5) | 权限审批与工具失败处理 |
| [s6](https://github.com/zhouNorth/KamaClaude/tree/s6) | 分层记忆与上下文压缩 |
| [s7](https://github.com/zhouNorth/KamaClaude/tree/s7) | Skill、MCP 与多 Agent 编排 |

当前版本的知识库配置见 [RAG 指南](RAG.md)，开发与运行命令见 [运行手册](RUNBOOK.md)，客户端通信格式见 [协议文档](WIRE_PROTOCOL.md)。
