# AEFS-Capstones

AEFS capstone 项目的 monorepo：一个项目一个顶层目录，评测互相隔离。

| 项目 | 一句话 | 入口 |
| --- | --- | --- |
| [`terminal-coding-agent`](projects/terminal-coding-agent/) | 终端里的 coding agent，在隔离 worktree 内改代码、跑命令、交 patch | [README](projects/terminal-coding-agent/README.md) · [PROJECT.md](projects/terminal-coding-agent/PROJECT.md) |
| [`multimodal-doc-qa`](projects/multimodal-doc-qa/) | 把文档页当图像检索，有界 agent 循环带引用作答，对照 vision-first 与 OCR-first | [README](projects/multimodal-doc-qa/README.md) · [PROJECT.md](projects/multimodal-doc-qa/PROJECT.md) |
| [`tiny-llm-pipeline`](projects/tiny-llm-pipeline/) | 从零训练 ~29M 中文小模型：预训练接龙 → SFT → DPO 对齐豆包体 | [README](projects/tiny-llm-pipeline/README.md) · [PROJECT.md](projects/tiny-llm-pipeline/PROJECT.md) |
| [`cyber-cricket`](projects/cyber-cricket/) | 游戏无关的 bot 竞技平台（C++20，零第三方依赖）；LLM 多智能体研发队是后续 feature | [PROJECT.md](projects/cyber-cricket/PROJECT.md)（**未落地**，README 待写） |

| 组件 | 一句话 | 导入名 |
| --- | --- | --- |
| [`repl-console`](components/repl-console/) | 共享 REPL：循环、基本显示，以及 `@` / `!` / `/` | `repl_console` |
| [`budget`](components/budget/) | 可选上限的花费账本 | `budget` |
| [`telemetry`](components/telemetry/) | OpenTelemetry 与 Langfuse 的共享安装 | `telemetry` |
| [`intent`](components/intent/) | 前置 chat/work 意图判定（一个结构化调用） | `intent` |

环境是仓库级共享的，统一入口是根目录 `./start.sh`（见 skill `start`）。新增项目或组件时在对应表里加一行；`./start.sh new` 的脚手架模板在 [`examples/hello-agent/`](examples/hello-agent/)，不是 capstone、不进上表。
