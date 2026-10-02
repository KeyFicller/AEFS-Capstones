# PROJECT.md — hello-agent

> 模板：每个项目根目录一份，作为项目特有信息的唯一来源。新项目直接复制本文件改写。

- **项目**：hello-agent / 所属 Phase：示例（不属于具体 Phase）
- **spec 链接**：（示例项目无 spec）

## 目标与范围

- 目标：示意「符合 rules 的完成态项目」长什么样。
- 不做：真实模型调用、真实沙箱、真实评测集。

## 架构

`plan -> act -> observe -> recover` 四段分离；假模型驱动；turn 上限熔断。

## 技术栈

Python 3.11+（本示例零依赖）。真实项目按 `20-python` 用
LangGraph / LangChain / PyTorch，模型走 OpenRouter。

## 指标与基线

- 指标：`pass@1`（示例中恒为 1.0）。
- 基线：hello-world（占位）。
- 数据集：示例 3 条任务；真实项目用真实数据集 + holdout。

## 预算

示例：`MAX_TURNS = 3`。真实项目在 PROJECT.md 写明轮数 / 上下文 / 成本上限。

## 交付物

- 命令：`uv run hello-agent run "say hello"`
- 结果：`eval/results.jsonl`

## 风险

- 示例刻意简化，不具备真实容错与安全边界；真实项目必须补齐 `40`/`50`。
