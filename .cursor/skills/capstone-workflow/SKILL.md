---
name: capstone-workflow
description: AEFS capstone 项目的完整生命周期工作流（接谱→设计→实现→加固→评测→交付）与每步的验证门槛。当开始、推进或交付本仓库任一 capstone 项目时使用。
---

# Capstone 工作流（对每个项目循环执行）

本仓库用于完成 AEFS（AI Engineering from Scratch）中选定的 capstone 项目。
**每个项目都走同一套工作流**；项目特有信息写在各自的 `PROJECT.md`。

## 生命周期（每步都有验证门槛）

1. **接谱**：读该项目 spec（`docs/en.md` + Build It / Ship It / rubric），产出项目整体的 `PROJECT.md`，并把范围拆成 features。
   → 验证：PROJECT.md 覆盖架构、栈、指标、预算、交付物，且 features 列全。
2. **设计**：一次一个 feature。先写 `docs/features/<slug>/design.md`，期间追问用户，行为和架构由用户拍板。用户确认后才写 `impl_plan.md`；方案若改了决策，同步改 design。见规则 `05-spec-driven`。
   → 验证：design 已经用户确认，且与 impl_plan 一致。
3. **实现**：按该 feature 的 impl_plan 做。要偏离先改 design 和 impl_plan。落地后回写 `PROJECT.md`。
   → 验证：每次改动后 `pytest` 绿，且 `PROJECT.md` 反映已落地的 feature。
4. **加固**：安全、预算、可观测性、错误恢复。
   → 验证：安全用例通过 + trace 完整。
5. **评测**：跑指标并与基线对比。
   → 验证：结果落盘且可复现。
6. **交付**：README + 演示命令 + 已知限制。
   → 验证：一条命令端到端跑通。

## Definition of Done

架构完整、可运行、可评测、可观测、有测试、有文档——细分要求见各专项规则。

## 反模式

- 复制 `code/` 示例当交付；只做 demo（硬编码结果、绕过真实流程、假数据）。
- 把某个项目的特有数值（工具集、预算、指标）硬编码进通用逻辑或通用规则。
- 未验证就宣称完成。

## 工作方式

- 一次聚焦一个项目；每个项目一个顶层目录，依赖与评测互相隔离。
- 项目特有内容一律来自 `PROJECT.md`，通用规则里的具体数字仅作示例。
- 假设与取舍显式记录；不确定先问，不要静默选择。
- 外科手术式改动：每行改动可追溯到当次需求（遵循 Karpathy 行为准则）。
