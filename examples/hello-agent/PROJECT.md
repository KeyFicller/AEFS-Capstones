# PROJECT.md — hello-agent

> 模板：每个项目根目录一份，作为项目特有信息的唯一来源。新项目直接复制本文件改写；栏目照抄既有项目的 `PROJECT.md`。

- **项目**：hello-agent / 所属 Phase：示例（不属于具体 Phase）
- **spec 链接**：（示例项目无 spec）

## 目标与范围

- 目标：给 `./start.sh new <name>` 提供最小可跑的项目骨架。
- 不做：任何真实功能。

## 架构

`core.py`（纯函数）+ `cli.py`（入口）。

## 技术栈

Python 3.11+，零依赖。真实项目按 `python` 规则用 LangGraph / LangChain，模型走 DeepSeek。

## 指标与基线

- 指标：`pass@1`（示例中恒为 1.0）。
- 基线：无。
- 数据集：示例 3 条；真实项目用真实数据集 + holdout。

## 预算

无。真实项目在此写明轮数 / 上下文 / 成本上限。

## 交付物

- 命令：`./start.sh run hello-agent world`
- 结果：`eval/results.jsonl`

## 风险

无。
