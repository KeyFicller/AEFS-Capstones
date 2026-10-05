# PROJECT.md — tiny-llm-pipeline

- **项目**：tiny-llm-pipeline / 所属 Phase：Phase19 / Capstone 07
- **spec 链接**：[https://aieng-zh.cn/lessons/19-capstone-projects/07-end-to-end-fine-tuning-pipeline/](https://aieng-zh.cn/lessons/19-capstone-projects/07-end-to-end-fine-tuning-pipeline/)
- **设计**：`docs/features/pretrain-sft-dpo/design.md`

## 目标与范围

- 目标：从零训练一个约 11M 的中文小模型，走通预训练接龙、SFT、DPO 对齐豆包体。行为与架构以 design 为准。
- 不做：量化、vLLM、K8s、安全检查、MOF model card、W&B、MinHash、PII。完整列表见 design 第 1.2 节。

## 架构

`config.py`（`ModelConfig` / `TrainConfig`）、`model.py`（`TinyLM`）、`tokenizer.py`、`data.py`、`train/pretrain.py`、`viz.py`（流程图与 `monitor.png`）已落地。CLI 有 `train-pretrain`、`flow`、`monitor`。SFT、DPO、偏好合成与评测尚未落地。

## 技术栈

Python 3.14.6，共享 venv。`torch`（MPS，fp32）、`tokenizers`、`numpy`、`langchain-deepseek`、`typer`、`rich`、`matplotlib`、`pytest`。不新增依赖。

## 指标与基线

预训练短跑（2026-10-05，MPS，fp32，batch 16 × seq 512，200 step，约 1.64M token）：loss 9.07 → 5.53（最低 4.90），val ppl **226.7**。随机初始化的交叉熵约 `ln(8192) ≈ 9.01`。这不是完整预训练的成绩，而且用的是拆段之前的语料。按 `<|im_start|>` / `<|im_end|>` 拆段后重写的 bin：train **99,723,098** token / 1,705,551 段，val **276,902** token / 8,570 段。这套 bin 还没有跑过训练。SFT / DPO 指标尚未实测。

## 预算

实测吞吐 **4713 tok/s**（同上短跑，约 6 分钟的稳态）。按这个速度，design 的 60 分钟墙钟大约是 **17M token**；默认 `--max-tokens 1e8` 大约 **5.9 小时**。默认预算没有改，等决定再钉。

同机、seq 512、fp32：batch **16** 吞吐最高。8 大约慢 5%，24/32 更慢；64 掉到内存颠簸，96 OOM。取 batch 的开销可以忽略。把序列改短不会更快。`TrainConfig.batch_size` 保持 16。

## 交付物

`train-pretrain` 可跑。第一次 Ctrl-C 会在当前步结束后写 `ckpt.pt`；`--resume <ckpt>` 从该步的数据游标和累计 token 接着训，不重放已经看过的窗口。200 step 的 ckpt 在 `artifacts/pretrain/ckpt.pt`。`eval/results.jsonl` 尚未接入。

## 风险

见 design 第 14 节。模型冒烟与 200 step 预训练短跑已通过。完整 token 预算尚未钉死。
