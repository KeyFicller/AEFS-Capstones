---
name: project-profile
description: 编写与维护项目根目录 PROJECT.md 的规范（必填栏目、规则与示例），它是项目特有信息的唯一来源。当开始新项目、读 spec、或需要项目特有配置（架构/栈/指标/预算）时使用。
---

# 项目档案 PROJECT.md

工作流保持通用；**项目特有的一切**只写在项目根目录的 `PROJECT.md`，
作为唯一事实来源。规则与 PROJECT.md 冲突时，以 PROJECT.md 为准。

## 必填栏目

- **项目**：名称 / 所属 Phase / spec 链接。
- **目标与范围**：要交付什么；明确不做什么。
- **架构**：spec 规定的分层与数据流（逐层列出）。
- **技术栈**：语言 / 框架 / 关键依赖（受全局硬基础 `20-python-stack` 约束）。
- **指标与基线**：评测指标、基线对象、数据集与规模。
- **预算**：轮数 / 上下文 / 成本等硬上限（若适用）。
- **交付物**：可运行命令、产物路径、结果/PR 文件。
- **风险**：已知坑与未决问题。

## 规则

- 开始任何项目前先读 spec 并写好 PROJECT.md；spec 更新则同步更新它。
- 通用规则里出现的具体值（工具清单、预算、指标）只是示例，实际以 PROJECT.md 为准。
- PROJECT.md 随项目一起提交；一个项目一份。

## 示例（Capstone 01）

架构 plan/act/observe/recover + hooks；6 工具；
预算 50 turns / 200k / $5；指标 SWE-bench Pro 30 题 pass@1 vs mini-swe-agent。
