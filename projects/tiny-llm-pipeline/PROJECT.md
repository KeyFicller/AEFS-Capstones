# PROJECT.md — tiny-llm-pipeline

- **项目**：tiny-llm-pipeline / 所属 Phase：Phase19 / Capstone 07
- **spec 链接**：[https://aieng-zh.cn/lessons/19-capstone-projects/07-end-to-end-fine-tuning-pipeline/](https://aieng-zh.cn/lessons/19-capstone-projects/07-end-to-end-fine-tuning-pipeline/)
- **设计**：`docs/features/pretrain-sft-dpo/design.md`

## 目标与范围

- 目标：从零训练一个约 11M 的中文小模型，走通预训练接龙、SFT、DPO 对齐豆包体。行为与架构以 design 为准。
- 不做：量化、vLLM、K8s、安全检查、MOF model card、W&B、MinHash、PII。完整列表见 design 第 1.2 节。

## 架构

`config.py`（`ModelConfig` / `TrainConfig`）、`model.py`（`TinyLM`）、`tokenizer.py`、`data.py`、`train/pretrain.py`、`train/sft.py`（SFT 循环与 held-out response-token ppl）、`train/dpo.py`（`dpo_loss` / `seq_logprob` / `train_dpo`，ref 为冻结的第二份拷贝）、`pref/synthesize.py`、`viz.py`（流程图与 `monitor.png`）已落地。CLI 有 `prepare`、`train-pretrain`、`synth-pref`、`train-sft`、`train-dpo`、`flow`、`monitor`。三段训练已在真实数据上跑通（`artifacts/smoke/`）；评测（`eval`）与推理 CLI（`gen` / `chat` / `compare`）尚未落地。

数据侧：`prepare` 一条命令从 `jingyaogong/minimind_dataset` 拉 `pretrain_hq.jsonl` / `sft_mini_512.jsonl`（走 HF 缓存，重跑只付一次下载），在 200MB 采样上训 `vocab=8192` 的 byte-level BPE，写出 `train.bin` / `val.bin` 与 `sft_train.jsonl` / `sft_holdout.jsonl` / `dpo_prompts.jsonl`。前处理与切分细节见 design 第 4、5 节。

## 技术栈

Python 3.12.13，共享 venv。`torch`（fp32）、`tokenizers`、`numpy`、`langchain-deepseek`、`typer`、`rich`、`matplotlib`、`pytest`。不新增依赖。

## 指标与基线

预训练短跑（2026-10-05，fp32，batch 16 × seq 512，200 step，约 1.64M token）：loss 9.07 → 5.53（最低 4.90），val ppl **226.7**。随机初始化的交叉熵约 `ln(8192) ≈ 9.01`。这不是完整预训练的成绩，而且用的是拆段之前的语料。按 `<|im_start|>` / `<|im_end|>` 拆段后重写的 bin：train **99,723,098** token / 1,705,551 段，val **276,902** token / 8,570 段。这套 bin 还没有跑过训练。SFT 与 DPO 已用超小批量打通接线（2026-10-06，mps，batch 16，各 3 步，产物在 `artifacts/smoke/`）：SFT（`--limit 32`）loss 8.36 → 7.45；DPO loss 0.6931 → 0.0015，`margin` -0.0 → 9.74。DPO 第 1 步的 `0.6931 = ln2` 是 policy 与 ref 同源时 `margin=0` 的解析值，不是拟合出来的，属接线正确的信号。**以上都是接线冒烟，不是训练成绩**：SFT 的 holdout ppl、DPO 的 `reward_margin` 与豆包体指标（design 第 8 节）仍需正式训练后实测。

## 预算

吞吐与 token 预算尚未在目标机器上实测。`--max-tokens` 默认 **1e8** 暂留，待目标机器上测出 tokens/s 后钉死正式 token 与墙钟。

`synth-pref` 的 API 预算是池子 1000 prompt × 2 次调用（估算 < ¥5）。到 404 对为止累计 **~1108 次调用**：首轮 200 prompt + 补量 70（合计 540），扩量轮 `--n 450` 再 235 prompt（+470），收尾 `--n 470` 再 49 prompt（+98，达标后手动停）。缺口全是模型拒答（"你是阿里员工吗"、"查财报"、"明天北京会下雨吗"），按 design 丢弃不截断。**每次重跑会把所有历史失败 prompt 再试一遍**（resume 只跳过已成功的），所以拒答越攒越多、扩量轮的调用量高于新增 prompt 数；拒答在 temperature 0.8 下高度可复现，重试基本不转正。

`TrainConfig.batch_size` 暂保持 **16**。batch 对吞吐与内存的影响依机器而异（含 OOM 阈值），换机后需重新扫一遍，不要沿用旧结论。

## 交付物

`prepare` 可跑：重跑复现出与既有产物**逐字节相同**的 `tokenizer_sample.txt` / `tokenizer.json` / `train.bin` / `val.bin`，并产出三份互不相交的 SFT 切分。`train-pretrain` 可跑。第一次 Ctrl-C 会在当前步结束后写 `ckpt.pt`；`--resume <ckpt>` 从该步的数据游标和累计 token 接着训，不重放已经看过的窗口。200 step 的 ckpt 在 `artifacts/pretrain/ckpt.pt`。

`synth-pref` 可跑（CLI 子命令，属数据流水线的一环）：读 `artifacts/data/dpo_prompts.jsonl`，写 `artifacts/prefs/prefs.jsonl`（`{"prompt","chosen","rejected","model","created_at"}`）。注意**它是流水线里唯一不能位级复现的一步**（要调 API，无 seed，见下节），所以 `prefs.jsonl` 必须随机器搬运。**追加式、默认续跑**：已在输出里的 prompt 跳过，中断重跑不重复计费；拒答 / 两侧相同 / 超 `max_seq_len=512` token 的对**丢弃并计数，不截断**。目标 **400** 对（理由见 design 第 5.3 节，2026-10-06 由 200 上调）；实跑到 **404 对**（`--n 450` 得 391，再 `--n 470` 补到 404 后停），404 个 prompt 全不重复，均值 chosen 125.2 字 / rejected 126.2 字，chosen 含豆包体标记 404/404，无超长对。

`train-sft` / `train-dpo` 已接入 CLI 并端到端跑通，交付命令（cwd 为仓库根，`$P=projects/tiny-llm-pipeline`）：

```bash
./start.sh run tiny-llm-pipeline train-pretrain --data $P/artifacts/data --out $P/artifacts/pretrain --max-steps 3
./start.sh run tiny-llm-pipeline train-sft --data $P/artifacts/data --base $P/artifacts/pretrain/ckpt.pt --out $P/artifacts/sft --max-steps 3 --limit 32
./start.sh run tiny-llm-pipeline train-dpo --data $P/artifacts/data --prefs $P/artifacts/prefs/prefs.jsonl --ref $P/artifacts/sft/ckpt.pt --out $P/artifacts/dpo --max-steps 3
```

2026-10-06 实跑结果（`--max-steps 3`，`/usr/bin/time` 计 8.3 s / 1.8 s / 1.6 s）：pretrain loss 9.07 → 8.31；SFT loss 8.36 → 7.45；DPO loss 0.6931 → 0.0015、`margin` -0.0 → 9.74。三段各自写出 `ckpt.pt` 与 `train_log.jsonl`，DPO 的 `ref` 文件全程只读未写。

`--max-steps` 是**绝对值**：`--resume` 时接着 ckpt 里的 `step` 数；fresh stage 从 0 起（base ckpt 的 `step` 属上一段，不继承，见 `test_step_counter_starts_fresh_from_base`）。`train-sft --limit` 控制读多少行（`sft_train.jsonl` 有 119 万行 / 1.2 GB，逐行流式读，不整份入内存）。

`eval/results.jsonl` 尚未接入。`gen` / `chat` / `compare` 尚未落地。

## 产物与跨机复现

2026-10-06 实测（Python 3.12.13 / torch 2.13.0 / tokenizers 0.23.2）：**数据侧产物全部位级可复现**，重跑得到逐字节相同的文件。

| 产物 | 代码可复现 | sha256 前 16 位 | 实测证据 |
| --- | --- | --- | --- |
| `tokenizer.json` | 是 | `9771982a59415853` | 从 `tokenizer_sample.txt` 重训 2 次，两次都与发布版同哈希 |
| `train.bin` | 是 | `6db23f6c0a626d40` | 全量重建（`--max-tokens 100_000_000`）同哈希、同 199,446,196 字节 |
| `val.bin` | 是 | `7c3ec8a099bec5d2` | 同上，553,804 字节 |
| `sft_train.jsonl` | 是 | `559807402e6a1daa` | 重建与发布版同哈希 |
| `sft_holdout.jsonl` | 是 | `ea0f920470b6d8fe` | 同上 |
| `dpo_prompts.jsonl` | 是 | `350a561c6ddb3bac` | 同上 |
| `pretrain_hq.jsonl` | 是（需下载） | `9801b0d2210c61c2` | `DATASET_REVISION` 钉在 `6b952cc5…`，1,669,750,047 字节 |
| `sft_mini_512.jsonl` | 是（需下载） | `475039fa9b80ad36` | 同上，1,232,540,940 字节 |
| `prefs.jsonl` | **否** | `424e5826e641cc19` | DeepSeek API 生成，temperature 0.8，无 seed 无缓存；215/240 的产出取决于接口行为 |
| `doubao-style-guide.md` | **否**（源文档） | `0c8f1c15893af605` | 手写纲要，不是任何代码的输出 |
| `pretrain/ckpt.pt` | 同类可复现，**非位级** | `4fa36c13e19ceaf4` | 同 seed/数据/代码可重训，但 MPS 与 CUDA 浮点路径不同 |

`prepare_pretrain` 的切分是 `val_docs, train_docs = kept[:n_val], kept[n_val:]`：**`train.bin` 从第 `n_val` 篇文档开始，不是第 0 篇**。做前缀比对要拿 `val.bin`（从第 0 篇起）。

**跨机搬运**：`artifacts/` 整体被根 `.gitignore` 第 35 行 `projects/*/artifacts/` 忽略，新机器 `git clone` **拿不到任何产物**。搬到云端时：

- **必须搬** `artifacts/prefs/`（`prefs.jsonl` + `doubao-style-guide.md`，共 200 KB）——代码生成不出 `prefs.jsonl`（要花钱调 API 且不位级复现），而 `doubao-style-guide.md` 是 `synth-pref --guide` 的默认值，缺了默认命令直接失败。
- `artifacts/data/` 4.2 GB 选择重生成或一起搬。重生成 = 重下 2.8 GB 原始语料 + 重跑 `prepare`（本机约 10 min），用上表哈希校验。
- `ckpt.pt` 不必搬，重训即可，且它本来就不是位级可复现的。

**依赖钉死的缺口**：`tokenizers` 与 `numpy` **没有**写进 `requirements.txt`，只由 `transformers==5.18.0` / `torch==2.13.0` 传递带入（当前 `tokenizers 0.23.2`、`numpy 2.5.3`）。`tokenizer.json` 的位级可复现依赖 BPE 训练器实现，换 tokenizers 版本可能改变 merges，连带 `train.bin` 与三份切分一起漂移——**换机后先重训 tokenizer 比对哈希，对齐再往下走**。另外 `requirements.txt` 是全仓库共用的，`./start.sh setup` 会把其他 6 个项目（含 `colpali-engine` / `streamlit` / `pytesseract` / `pymupdf`）连 `-e` 安装一起拖进来；云端建议按需最小安装。Linux GPU 机器还需自己确认 torch 拿到的是 CUDA wheel（`start.sh` 只在 Windows 走 `--torch-backend=cu126`）。

## 风险

见 design 第 14 节。模型冒烟与 200 step 预训练短跑已通过。完整 token 预算尚未钉死。
