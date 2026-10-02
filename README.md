# AEFS-Capstones

AEFS capstone 项目的 monorepo。每个 capstone 一个顶层目录，依赖与评测互相隔离。

## 项目导航

`projects/` 下一个 capstone 一个顶层目录，`examples/` 放脚手架模板。新增项目请在下面加一行。

| 项目 | 说明 | 入口 |
| --- | --- | --- |
| [`terminal-coding-agent`](projects/terminal-coding-agent/) | Capstone 01 / Phase 19：终端原生 coding agent。在隔离 worktree 内读改代码、跑命令、交 patch；评测用手写 Harbor 任务集通过率 | [README](projects/terminal-coding-agent/README.md) · [PROJECT.md](projects/terminal-coding-agent/PROJECT.md) |
| [`multimodal-doc-qa`](projects/multimodal-doc-qa/) | Capstone 04 / Phase 19：Agentic RAG 文档问答。ColQwen2.5 后期交互检索 + 有界 agent 循环（规划/多轮/自省/自检）+ 托管 VLM 带引用合成 + 证据区域查看器；对 OCR-first 文本基线 | [PROJECT.md](projects/multimodal-doc-qa/PROJECT.md) |
| [`hello-agent`](examples/hello-agent/) | 脚手架模板（非 capstone）：演示符合 `.cursor/rules/` 的完成态项目长什么样 | [README](examples/hello-agent/README.md) · [PROJECT.md](examples/hello-agent/PROJECT.md) |
