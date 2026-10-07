# tiny-llm-pipeline

- **项目**：tiny-llm-pipeline / 所属 Phase：Phase19 / Capstone 07
- **spec**：[Capstone 07 — 端到端微调流水线](https://aieng-zh.cn/lessons/19-capstone-projects/07-end-to-end-fine-tuning-pipeline/)
- **状态**（2026-10-07）：三段已在真实数据上训练完——预训练 34500 步（val ppl **11.8**）、SFT 5000 步（留出集回答 ppl **6.34**，微调前的 `pretrain-30m` 基线 **38.15**）、DPO 104 步（Δ = `logπ_chosen − logπ_rejected` 由 **−197.8** 升到 **+331.9**，Δ>0 的偏好对 **22% → 64%**）。`prepare` 位级可复现；`eval/sample.py` 与 `eval/serve.py` 已落地；单测 `77 passed`。**未落地**：`eval/results.jsonl`、`gen` / `chat` / `compare` CLI。**交付配置**：DPO `--beta 0.01 --epochs 4 --lr 3e-5`，且**生成时必须开 `repetition_penalty`（1.3–1.5）——它是交付的一部分，不是可选的润色**。
- **产物位置**：权重与日志在 `checkpoints/{pretrain,sft,dpo}/`（`ckpt.pt` + `train_log.jsonl`，pretrain / SFT 另有 `monitor.png`），tokenizer 在 `checkpoints/tokenizer/tokenizer.json`；CLI 默认的输入输出目录是 `artifacts/`。**两者都被根 `.gitignore` 忽略**（`checkpoints/`、`projects/*/artifacts/*`），新机器 `git clone` 拿不到。唯一强制入库的 artifacts 是 `artifacts/prefs/`——`prefs.jsonl` 靠调 API 生成、**不可位级复现**，必须随机器搬运；`doubao-style-guide.md` 是 `synth-pref --guide` 的默认值，缺了默认命令直接失败。
- **权重托管**：三段权重公开在 [HF `KeyFicller/tiny-llm-pipeline-29m`](https://huggingface.co/KeyFicller/tiny-llm-pipeline-29m)，布局与本项目一致，`Bundle.load(<snapshot>)` 可直接加载、无需改代码。**已剥离 `optimizer`**（因此不能 `--resume` 续训），权重与本地训练产物**逐张量相同**、回下载 sha256 一致，端到端生成结果一致。推/拉用 `checkpoints/sync_ckpt.py push` / `pull`。
- **`checkpoints/sync_ckpt.py`**：它是 `checkpoints/` 下**唯一入库**的文件（`.gitignore` 为它单开一条 `!` 例外，其余 ckpt / 日志 / `weights/` / `.sync_manifest.json` 全部忽略），所以脚本跟着仓库走、权重不跟着。`push` 只上传剥离 `optimizer` 后的权重（`checkpoints/weights/`），`pull` 拉到 `checkpoints/`。需要它而不是手敲 `hf upload` 的两个原因：①`pull` 会**拒绝覆盖含 `optimizer` 的可续训完整 ckpt**（Hub 上是瘦身版，覆盖即永久失去 `--resume`），要覆盖得显式 `--force` 或改写到 `--out`；②`torch.save` **字节不稳定**（zip 里带了每次写都不同的 `serialization_id`），重新剥离永远得不到已上传的那串字节，故 `push` 靠 `checkpoints/.sync_manifest.json` 记源 ckpt 的 sha256，源没变就沿用已暂存字节——否则每次 push 都造一个内容等价的空提交。源本就是瘦身版时逐字节复制，故新机器 `pull` 后 `push` 也是干净空操作。`--stages` / `--dry-run` 可分别选段与空跑。

## 目标与范围

**目标**：从零训练一个约 29M 的中文小模型，走通预训练接龙 → SFT → DPO 对齐豆包体。

- **模型**：`ModelConfig` 对齐 MiniMind2-Small——词表 **6400**、宽度 512、8 层、8 个 query 头、2 个 KV 头；FFN 宽度取仓库公式 `ceil(512 * pi / 64) * 64 = 1664`；绑定词嵌入后 **28,975,616** 参数。训练窗口 `max_seq_len = 768`（`pretrain_t2t_mini` 的推荐长度）。
- **预训练数据**：`prepare` 从 `jingyaogong/minimind_dataset` 拉 `pretrain_hq.jsonl` / `sft_mini_512.jsonl`（走 HF 缓存，重跑只付一次下载），在 200MB 采样上训词表 6400 的 byte-level BPE，再打包 `train.bin` / `val.bin`。**必须钉 `DATASET_REVISION`**：该 dataset 的 `main` 已把这两个文件换成 `pretrain_t2t*` 系列，不钉旧 commit 会在新机器上 404。
- **SFT 数据**：源文件里前后相邻的问答**没有上下文关系**，`prepare_sft` 先把每条记录拆成独立的一问一答，再切出**互不相交**的三份——`sft_train.jsonl` / `sft_holdout.jsonl` / `dpo_prompts.jsonl`。
- **偏好对**：`synth-pref` 读 `dpo_prompts.jsonl`，写 `{prompt, chosen, rejected, model, created_at}`。**丢弃不截断**：拒答、两侧相同、`prompt + 回答` 超过 `max_seq_len = 768` token 的对一律丢掉并计数，同时写 `<stem>.skipped.jsonl` 旁挂文件。目标 **400** 对。
- **不做**：量化、vLLM、K8s、安全检查、MOF model card、W&B、MinHash 去重、PII 过滤。

## 架构

均在 `src/tiny_llm_pipeline/` 下：

| 路径 | 职责 |
| --- | --- |
| `config.py` | `ModelConfig` / `TrainConfig` / `PRETRAIN_TOKENS`，所有可调项 |
| `model.py` | `TinyLM`（tied embedding） |
| `tokenizer.py` | byte-level BPE 训练与加载 |
| `data.py` | 预训练打包 bin；SFT 多轮拆单轮 + 三份切分 |
| `train/pretrain.py` | 预训练循环 |
| `train/sft.py` | SFT 循环与 held-out response-token ppl |
| `train/dpo.py` | `dpo_loss` / `seq_logprob` / `train_dpo`；`ref` 是**冻结的第二份拷贝** |
| `pref/synthesize.py` | 偏好对合成与过滤 |
| `generate.py` | 三段推理与读数口径（`seed_for` / `stats` / `GenSettings` / `Bundle`） |
| `viz.py` | 流程图与 `monitor.png` |
| `eval/sample.py` | 离线抽样 + 自包含 HTML |
| `eval/serve.py` | 本地对照页（`GET /` + `POST /ask`）；与 `sample.py` 共用 `generate.py` |
| `cli.py` | `prepare` / `synth-pref` / `train-pretrain` / `train-sft` / `train-dpo` / `flow` / `monitor` |

**图产物**：`graph.png` 由 `flow` 子命令生成，拓扑改了重跑。README 只放这一张图。

## 技术栈

Python 3.12.13，共享 venv。`torch`（fp32；设备按 CUDA → MPS → CPU 的顺序选取）、`tokenizers`、`numpy`、`langchain-deepseek`、`typer`、`rich`、`matplotlib`、`pytest`。不新增依赖。

## 交付物

**训练（已交付）**

- `prepare` 的产物**位级可复现**：同版本 `tokenizers` 下重跑得到与既有产物**逐字节相同**的 `tokenizer.json` / `train.bin` / `val.bin` 与三份切分。**但这份确定性依赖 BPE 训练器实现**——`tokenizers` 与 `numpy` 只由 `transformers` / `torch` 传递带入、没写进 `requirements.txt`，换版本可能改变 merges，连带 bin 与切分一起漂移，**换机后先重训 tokenizer 比对哈希再往下走**。切分口诀 `val_docs, train_docs = kept[:n_val], kept[n_val:]` → **`train.bin` 从第 `n_val` 篇文档开始**，做前缀比对要拿 `val.bin`（从第 0 篇起）。
- **存盘**：每 **500** 步写 `ckpt.pt`（优化器 / 累计 token / 行游标）。**`ckpt.pt` 每步覆盖同一路径**，最优权重可能被末步盖掉——SFT 靠 `best.pt` 兜（见下）。
- **预训练**：`--val-every` 默认 **-1**（关）。`--resume` 从该步的数据游标与累计 token 接着训，不重放看过的窗口。
- **SFT**：存盘同预训练，另有 `monitor.png`（每 10 步覆盖）与 `--val-every` 默认 **1000**（按该步从 holdout 抽 5 条贪心作答，不重复固定的前 5 条）。**留出困惑度最低的那次另存 `best.pt`，也是返回值**——续跑时从日志恢复已有最优值，更差的步盖不掉。`--resume` 从行游标接着训，并截掉日志里步数更大的行。
- **DPO**：只写 `ckpt.pt`（无 `monitor.png` / `best.pt`）。`--ref` 全程只读未写。
- **`--max-steps` 是绝对值**：`--resume` 时接着 ckpt 里的 `step`；fresh stage 从 0 起（base ckpt 的 `step` 属上一段，不继承）。
- **`--lr` 默认** `SFT_LR = 2e-4` / `DPO_LR = 1e-5`；交付用的是 DPO `--lr 3e-5`。`--beta` 与 `--epochs` 也必须显式给——默认 `beta = 0.1` 落在「通顺但没风格」的欠训点。
- **`--max-steps` 决定余弦视野**：`--resume` 时只调大它会把学习率重新抬高（5000 → 7000 时 `2.0e-5` 跳回 `5.4e-5`，留出困惑度 **6.34 → 8.92**）。要延长训练得**保持原视野**或显式给更长的计划。DPO 同理，视野由 `--epochs × 每 epoch 步数` 或 `--max-steps` 决定——这是同一 `beta` 下不同 `epochs` 结果差很多的原因。

**`synth-pref`（数据流水线一环）**：**追加式、默认续跑**——已在输出里的 prompt 跳过，中断重跑不重复计费；**重跑会把所有历史失败 prompt 再试一遍**（resume 只跳过已成功的），而拒答在 `temperature 0.8` 下高度可复现、重试基本不转正，所以拒答越攒越多、扩量轮的调用量高于新增 prompt 数。**它是流水线里唯一不可位级复现的一步**（调 API，无 seed 无缓存）。

**推理与查看（已落地）**

- `eval/sample.py`：离线抽样 + 自包含 HTML。种子由 `(seed, stage, rep_pen, prompt, index)` 派生，同一批可复现、与执行顺序无关。
- `eval/serve.py`：本地对照页，三段并排，每栏带 `chars` / `dist4` / `tempo` 三项读数。只绑 `127.0.0.1`、无鉴权、**不落盘**。默认 `rep_pen=1.5`、`temperature=0`（贪心）、`max_new=64`。
- 两者共用 `generate.py`，重构到共用之前后同一参数同一种子产出的 HTML **逐字节相同**。同一问在 `temperature=0.8` 下 DPO 栏出现 U+FFFD——**采样只掉连贯性、不加内容**，故页面默认仍是贪心。

**未落地**：`eval/results.jsonl`、`gen` / `chat` / `compare`。落地时 **`gen` 必须暴露 `repetition_penalty`（默认约 1.3）**，否则交付的 DPO 权重会复读。
