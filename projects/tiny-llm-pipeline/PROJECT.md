# PROJECT.md — tiny-llm-pipeline

- **项目**：tiny-llm-pipeline / 所属 Phase：Phase19 / Capstone 07
- **spec 链接**：[https://aieng-zh.cn/lessons/19-capstone-projects/07-end-to-end-fine-tuning-pipeline/](https://aieng-zh.cn/lessons/19-capstone-projects/07-end-to-end-fine-tuning-pipeline/)
- **设计**：`docs/features/pretrain-sft-dpo/design.md`

## 目标与范围

- 目标：从零训练一个约 11M 的中文小模型，走通预训练接龙、SFT、DPO 对齐豆包体。行为与架构以 design 为准。
- 不做：量化、vLLM、K8s、安全检查、MOF model card、W&B、MinHash、PII。完整列表见 design 第 1.2 节。

## 架构

`config.py`（`ModelConfig` / `TrainConfig`）、`model.py`（`TinyLM`）、`tokenizer.py`、`data.py`、`train/pretrain.py`、`viz.py`（流程图与 `monitor.png`）已落地。CLI 有 `prepare`、`train-pretrain`、`flow`、`monitor`。SFT、DPO、偏好合成与评测尚未落地。

数据侧：`prepare` 一条命令从 `jingyaogong/minimind_dataset` 拉 `pretrain_hq.jsonl` / `sft_mini_512.jsonl`（走 HF 缓存，重跑只付一次下载），在 200MB 采样上训 `vocab=8192` 的 byte-level BPE，写出 `train.bin` / `val.bin` 与 `sft_train.jsonl` / `sft_holdout.jsonl` / `dpo_prompts.jsonl`。前处理与切分细节见 design 第 4、5 节。

## 技术栈

Python 3.14.6，共享 venv。`torch`（fp32）、`tokenizers`、`numpy`、`langchain-deepseek`、`typer`、`rich`、`matplotlib`、`pytest`。不新增依赖。

## 指标与基线

预训练短跑（2026-10-05，fp32，batch 16 × seq 512，200 step，约 1.64M token）：loss 9.07 → 5.53（最低 4.90），val ppl **226.7**。随机初始化的交叉熵约 `ln(8192) ≈ 9.01`。这不是完整预训练的成绩，而且用的是拆段之前的语料。按 `<|im_start|>` / `<|im_end|>` 拆段后重写的 bin：train **99,723,098** token / 1,705,551 段，val **276,902** token / 8,570 段。这套 bin 还没有跑过训练。SFT / DPO 指标尚未实测。

## 预算

吞吐与 token 预算尚未在目标机器上实测。`--max-tokens` 默认 **1e8** 暂留，待目标机器上测出 tokens/s 后钉死正式 token 与墙钟。

`TrainConfig.batch_size` 暂保持 **16**。batch 对吞吐与内存的影响依机器而异（含 OOM 阈值），换机后需重新扫一遍，不要沿用旧结论。

## 交付物

`prepare` 可跑：重跑复现出与既有产物**逐字节相同**的 `tokenizer_sample.txt` / `tokenizer.json` / `train.bin` / `val.bin`，并产出三份互不相交的 SFT 切分。`train-pretrain` 可跑。第一次 Ctrl-C 会在当前步结束后写 `ckpt.pt`；`--resume <ckpt>` 从该步的数据游标和累计 token 接着训，不重放已经看过的窗口。200 step 的 ckpt 在 `artifacts/pretrain/ckpt.pt`。`eval/results.jsonl` 尚未接入。

## 风险

见 design 第 14 节。模型冒烟与 200 step 预训练短跑已通过。完整 token 预算尚未钉死。
