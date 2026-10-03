# AEFS-Capstones

AEFS capstone 项目的 monorepo：一个项目一个顶层目录，依赖与评测互相隔离。

| 项目 | 一句话 | 入口 |
| --- | --- | --- |
| [`terminal-coding-agent`](projects/terminal-coding-agent/) | 终端里的 coding agent，在隔离 worktree 内改代码、跑命令、交 patch | [README](projects/terminal-coding-agent/README.md) · [PROJECT.md](projects/terminal-coding-agent/PROJECT.md) |
| [`multimodal-doc-qa`](projects/multimodal-doc-qa/) | 把文档页当图像检索，有界 agent 循环带引用作答，对照 vision-first 与 OCR-first | [README](projects/multimodal-doc-qa/README.md) · [PROJECT.md](projects/multimodal-doc-qa/PROJECT.md) |
| [`hello-agent`](examples/hello-agent/) | `./start.sh new` 的脚手架模板（非 capstone） | [README](examples/hello-agent/README.md) · [PROJECT.md](examples/hello-agent/PROJECT.md) |

环境是仓库级共享的，统一入口是根目录 `./start.sh`（见 skill `start`）。新增项目时在表里加一行。
