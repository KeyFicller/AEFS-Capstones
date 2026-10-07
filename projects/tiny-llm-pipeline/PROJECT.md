# PROJECT.md — tiny-llm-pipeline

- **项目**：tiny-llm-pipeline / 所属 Phase：Phase19 / Capstone 07
- **spec 链接**：[https://aieng-zh.cn/lessons/19-capstone-projects/07-end-to-end-fine-tuning-pipeline/](https://aieng-zh.cn/lessons/19-capstone-projects/07-end-to-end-fine-tuning-pipeline/)
- **设计**：`docs/features/pretrain-sft-dpo/design.md`

## 目标与范围

- 目标：从零训练一个约 29M 的中文小模型，走通预训练接龙、SFT、DPO 对齐豆包体。行为与架构以 design 为准。当前 `ModelConfig` 对齐 MiniMind2-Small：词表 6400、宽度 512、8 层、8 个 query 头、2 个 KV 头，FFN 宽度取仓库公式 `ceil(512 * pi / 64) * 64 = 1664`，绑定词嵌入后 **28,975,616** 参数。训练窗口 `max_seq_len=768`，对应 `pretrain_t2t_mini` 的推荐长度。
- 不做：量化、vLLM、K8s、安全检查、MOF model card、W&B、MinHash、PII。完整列表见 design 第 1.2 节。

## 架构

`config.py`（`ModelConfig` / `TrainConfig`）、`model.py`（`TinyLM`）、`tokenizer.py`、`data.py`、`train/pretrain.py`、`train/sft.py`（SFT 循环与 held-out response-token ppl）、`train/dpo.py`（`dpo_loss` / `seq_logprob` / `train_dpo`，ref 为冻结的第二份拷贝）、`pref/synthesize.py`、`viz.py`（流程图与 `monitor.png`）、`generate.py`（三段推理与读数口径，`seed_for` / `stats` / `GenSettings` / `Bundle`）已落地。`eval/sample.py`（离线抽样 + 自包含 HTML）与 `eval/serve.py`（本地对照页，`GET /` + `POST /ask`）已落地，两者共用 `generate.py`。CLI 有 `prepare`、`train-pretrain`、`synth-pref`、`train-sft`、`train-dpo`、`flow`、`monitor`。三段训练已在真实数据上跑通（接线冒烟在 `artifacts/smoke/`；预训练 / SFT / DPO 的正式权重分别在 `artifacts/pretrain-30m/`、`artifacts/sft-30m/`、`artifacts/dpo-30m/`）；评测（`eval`）与推理 CLI（`gen` / `chat` / `compare`）尚未落地。

数据侧：`prepare` 一条命令从 `jingyaogong/minimind_dataset` 拉 `pretrain_hq.jsonl` / `sft_mini_512.jsonl`（走 HF 缓存，重跑只付一次下载），在 200MB 采样上训 `vocab=8192` 的 byte-level BPE，写出 `train.bin` / `val.bin` 与 `sft_train.jsonl` / `sft_holdout.jsonl` / `dpo_prompts.jsonl`。前处理与切分细节见 design 第 4、5 节。

## 技术栈

Python 3.12.13，共享 venv。`torch`（fp32；设备按 CUDA、MPS、CPU 的顺序选取）、`tokenizers`、`numpy`、`langchain-deepseek`、`typer`、`rich`、`matplotlib`、`pytest`。不新增依赖。

## 指标与基线

11M（5 层）完整预训练（2026-10-05，fp32，batch 16 × seq 512，1.000e8 token，12231 step，约 13.5 分钟）：末步 loss **3.31**，val ppl **40.47**。随机初始化的交叉熵约 `ln(8192) ≈ 9.01`。checkpoint 在 `artifacts/pretrain/ckpt.pt`。

20M 这次把 `pretrain_hq.jsonl` 按同样规则打满：train **348,135,922** token，val **567,293** token，合计 **348,703,215**。Chinchilla 预算是 **409,121,280** token（20 × 20,456,064），独立语料少约 15%，多出来的步会在 bin 上再循环。该次停在第 50000 步（预算 50040），累计 **408,800,000** token，val ppl **28.1**。checkpoint 在 `artifacts/pretrain-20m/ckpt.pt`。换词表后这份 bin 和 tokenizer 会被 30M 的准备过程覆盖，20M 权重不再对得上新的 `tokenizer.json`。

30M 这次改用 MiniMind 当前推荐的轻量预训练语料 `pretrain_t2t_mini.jsonl`（dataset `main`，1,241,043,656 字节，从 ModelScope 国内 CDN 下载）。词表 6400 的 byte-level BPE 训完后，train **313,991,763** token，val **1,152,847** token，合计 **315,144,610**。Chinchilla 预算是 **579,512,320** token（20 × 28,975,616），独立语料大约只够 0.54 个预算，多出来的步会在 bin 上再循环。停在第 34500 步（预算 47223），累计 **423,384,000** token，val ppl **11.8**。checkpoint 在 `artifacts/pretrain-30m/ckpt.pt`。

SFT 数据：`sft_t2t_mini.jsonl`（从 ModelScope 下载，1,739,201,170 字节）里前后相邻的问答没有上下文关系，`prepare_sft` 先把每条记录拆成独立的一问一答再做切分。全量拆完 train **1,124,399** 条 / holdout **200** 条 / DPO 提示 **1000** 条，每条都是两轮。

SFT 第一次正式训练（2026-10-06）**失败，失败模式是过拟合**：从 `artifacts/pretrain-30m/ckpt.pt` 出发，`--limit` 换成的每 3 条取 1 条（374,799 条）、1 个 epoch、23,425 步，学习率沿用预训练的 `1e-3`。训练 loss 2.27 → **0.53**，留出集回答困惑度在第 **4,500** 步触底 **11.85** 后恶化到末步 **36.56**。产物连日志存在 `checkpoints/sft-30m-overfit/`，失败说明见该目录的 `FAILURE.md`。根因是峰值过高；另外 `ckpt.pt` 每步覆盖同一路径，第 4,500 步的最优权重已被末步盖掉。修正见「交付物」一节。

SFT 第二次训练（同一天，`--lr 2e-4`，`--max-steps 5000`，约 7 分钟）：留出集回答困惑度从第 500 步的 **7.50** 单调降到第 5000 步的 **6.34**，训练 loss 1.80 → 1.54（末步 5.04 是单批噪声）。同一份 200 行留出集上，未微调的 `pretrain-30m` 基线是 **38.15**，SFT 后 **6.34**，约为基线的 1/6。这就是保留的交付权重：`artifacts/sft-30m/ckpt.pt`（step 5000，6,191,854 个回答 token）。

曾试过在这一版上续训到 7000 步，**结果作废**：`--resume` 时把 `--max-steps` 从 5000 改成 7000，`lr_at` 只按新的视野算余弦，学习率在第 5000 步被重新抬到 `5.4e-5`（原本是 `2.0e-5`），困惑度随即从 6.34 升到 7.99 / 8.45 / 8.71 / **8.92**（7000 步）。`best.pt` 挡住了回归，但**延长总步数会重置 schedule** 这点得记住：要加训应当保持原视野只走退火尾巴，或重设一条更长的余弦，不能直接改 `--max-steps`。

要留意：两次的 holdout 都是同一份 200 行文件，但行型从多轮变成了单轮，所以 11.85 / 36.56 与 7.50 / 6.34 不是严格同口径的对比；2e-4 的结论主要来自它自己这条单调下降的曲线。

SFT 与 DPO 已用超小批量打通接线（2026-10-06，mps，batch 16，各 3 步，产物在 `artifacts/smoke/`）：SFT（`--limit 32`）loss 8.36 → 7.45；DPO loss 0.6931 → 0.0015，`margin` -0.0 → 9.74。DPO 第 1 步的 `0.6931 = ln2` 是 policy 与 ref 同源时 `margin=0` 的解析值，不是拟合出来的，属接线正确的信号。**以上 SFT / DPO 都是接线冒烟，不是训练成绩**：正式成绩见下。

DPO 正式训练（2026-10-06）：从 `artifacts/sft-30m/ckpt.pt` 出发，policy 与 ref 同源，偏好对是 `artifacts/prefs/prefs.jsonl` 的 **首轮 404 对**（即现 822 对文件的前 404 行；batch 16 → 每 epoch 26 步）。交付配置 `--beta 0.01 --epochs 4 --lr 3e-5`，共 **104 步 / 374,855 个计分 token / 33.4 s**，末步 loss 0.0234、日志 `margin` 5.20。

**日志里的 `margin` 是「比 ref 更偏好 chosen 的量」，不是「policy 已经偏好 chosen」**（`dpo_loss` 的 docstring 已按这个含义改准）。对 404 对逐条重算 `Δ = logπ_chosen − logπ_rejected`：SFT **−197.8** → DPO **+331.9**，改善 **+529.7**，`Δ>0` 的偏好对从 **22%** 升到 **64%**。

调参扫描（11 个 run，`beta ∈ {0.1, 0.01, 0.005, 0.003}` × `epochs ∈ {3,4,6,8,10,12}` × `lr ∈ {1e-5, 3e-5}`；各 run 的 ckpt 已删，`train_log.jsonl` 与全部生成样例留档在 `artifacts/dpo-sweep/`）：

| beta | epochs | lr | Δ | Δ>0 | greedy（`repetition_penalty=1.0`） |
| --- | --- | --- | --- | --- | --- |
| 0.1 | 3 | 1e-5 | 104 | 31% | 通顺，**无风格** |
| 0.01 | 3 | 1e-5 | 183 | 36% | 通顺，**无风格** |
| 0.01 | 6 | 1e-5 | 318 | 49% | 基本通顺，风格弱 |
| 0.01 | 8 | 1e-5 | 408 | 55% | 风格出现，开始复读 |
| 0.01 | 4 | **3e-5** | **530** | **64%** | 复读 ← **交付** |
| 0.01 | 10 | 1e-5 | 549 | 66% | 复读 |
| 0.01 | 12 | 1e-5 | 840 | 82% | 复读 |
| 0.01 | 6 | 3e-5 | 1101 | 87% | 复读 |
| 0.005 | 12 | 1e-5 | 1542 | 89% | 严重复读 |
| 0.003 | 12 | 1e-5 | 1978 | 91% | 严重复读 |
| 0.01 | 12 | 3e-5 | 2161 | 99% | 风格最全，复读最重 |

**豆包体是学得到的，但没有一个点同时做到「风格明显」和「greedy 不复读」。** `Δ ≥ 318` 起输出开始掉进「最不绕弯、最不绕弯…」的循环；`Δ ≈ 500–1100` 这一段要生成时开 `repetition_penalty` 才可读——`1.0` 循环、`1.15` 约一半仍循环、`1.3–1.5` 不再循环但内容开始游离（编书名、编经历）。**`repetition_penalty` 是交付的一部分，不是可选的润色。**

根因不在 DPO 而在数据：chosen 的开头高度模板化——404 条里 **85 条（21%）前 6 字完全相同**（「我用最直白」），骨架清一色「我用最X、最Y、最Z的方式…」。模型学到这个模板后接不上正文，只能原地复读。风格纲要第 1 条把前摇设计成「可无限堆叠」，代价就是 chosen 之间开头同质。

**加量无效已被实测证明（2026-10-06）**：`prefs.jsonl` 由 404 追加到 **822** 对（新增 418 对，prompt 全不重复），但新增对与旧对同分布（首 6 字最高占比 18.7% vs 21.0%，以「我用最」开头 41.1% vs 45.3%），没触碰开头同质的根因。用 822 对、交付 SFT 作 ref 重训（`beta 0.01 / lr 3e-5`）：`epochs 2`（104 步，与交付同视野）`margin_avg` 5.20，同一份旧 404 子集上 Δ **+353.4**、Δ>0 **63.9%**，相对交付的 +331.8 / 64.4% 无可测差异；`epochs 3`（156 步）Δ 跳到 **+947**、Δ>0 89%；`epochs 4`（208 步）Δ **+1380**、Δ>0 95%，贪心复读比交付更重。**现有数据上没有任何操作点能超过交付版，多训比少训更差。** 复现用的 822 系 run 产物（ckpt 与样例）已按决定删除、未留档；交付仍是 404 那版 `artifacts/dpo-30m/ckpt.pt`，未替换。

## 预算

这台机器（2026-10-05，RTX 4090 49 GB，CUDA 12.6，fp32，seq 512，预热 3 步后计 10 步，随机 token，含反传与 AdamW）：

| batch | tok/s | 峰值显存 |
| --- | --- | --- |
| 8 | 208835 | 1.3 GB |
| 16 | 235395 | 2.5 GB |
| 32 | 239771 | 4.8 GB |
| 64 | **240419** | 9.5 GB |
| 128 | 240095 | 18.8 GB |
| 256 | 239776 | 37.5 GB |
| 512 及以上 | OOM | |

`synth-pref` 的 API 预算是池子 1000 prompt × 2 次调用（估算 < ¥5）。到 404 对为止累计 **~1108 次调用**：首轮 200 prompt + 补量 70（合计 540），扩量轮 `--n 450` 再 235 prompt（+470），收尾 `--n 470` 再 49 prompt（+98，达标后手动停）。缺口全是模型拒答（"你是阿里员工吗"、"查财报"、"明天北京会下雨吗"），按 design 丢弃不截断。**每次重跑会把所有历史失败 prompt 再试一遍**（resume 只跳过已成功的），所以拒答越攒越多、扩量轮的调用量高于新增 prompt 数；拒答在 temperature 0.8 下高度可复现，重试基本不转正。

batch **64** 最高，但 32–256 与它相差不到 0.3%。默认 `TrainConfig.batch_size` 仍是 **16**。正式预训练 token 预算钉为 `PRETRAIN_TOKENS = 579_512_320`，对应 20 token/参数。上面的 tok/s 是 11M 模型上测的，不能直接套到 30M。

## 交付物

`prepare` 可跑：重跑复现出与既有产物**逐字节相同**的 `tokenizer_sample.txt` / `tokenizer.json` / `train.bin` / `val.bin`，并产出三份互不相交的 SFT 切分。`train-pretrain` 可跑。第一次 Ctrl-C 会在当前步结束后写 `ckpt.pt`；`--resume <ckpt>` 从该步的数据游标和累计 token 接着训，不重放已经看过的窗口。200 step 的 ckpt 在 `artifacts/pretrain/ckpt.pt`。

`synth-pref` 可跑（CLI 子命令，属数据流水线的一环）：读 `artifacts/data/dpo_prompts.jsonl`，写 `artifacts/prefs/prefs.jsonl`（`{"prompt","chosen","rejected","model","created_at"}`）。注意**它是流水线里唯一不能位级复现的一步**（要调 API，无 seed，见下节），所以 `prefs.jsonl` 必须随机器搬运。**追加式、默认续跑**：已在输出里的 prompt 跳过，中断重跑不重复计费；拒答 / 两侧相同 / 超 `max_seq_len=512` token 的对**丢弃并计数，不截断**。目标 **400** 对（理由见 design 第 5.3 节，2026-10-06 由 200 上调）；首轮实跑到 **404 对**（`--n 450` 得 391，再 `--n 470` 补到 404 后停），404 个 prompt 全不重复，均值 chosen 125.2 字 / rejected 126.2 字，chosen 含豆包体标记 404/404，无超长对。**2026-10-06 又追加 418 对到 822**（commit `fa06729`）：822 个 prompt 全不重复，均值 chosen 124.6 字 / rejected 127.6 字，新增对与首轮同分布（见「指标与基线」）。**交付的 `dpo-30m` 只用首轮 404 对训练**，822 对重训的结论见上。

`train-sft` / `train-dpo` 已接入 CLI 并端到端跑通，交付命令（cwd 为仓库根，`$P=projects/tiny-llm-pipeline`）：

```bash
./start.sh run tiny-llm-pipeline train-pretrain --data $P/artifacts/data --out $P/artifacts/pretrain --max-steps 3
./start.sh run tiny-llm-pipeline train-sft --data $P/artifacts/data --base $P/artifacts/pretrain/ckpt.pt --out $P/artifacts/sft --max-steps 3 --limit 32
./start.sh run tiny-llm-pipeline train-dpo --data $P/artifacts/data --prefs $P/artifacts/prefs/prefs.jsonl --ref $P/artifacts/sft/ckpt.pt --out $P/artifacts/dpo --max-steps 3
```

2026-10-06 实跑结果（`--max-steps 3`，`/usr/bin/time` 计 8.3 s / 1.8 s / 1.6 s）：pretrain loss 9.07 → 8.31；SFT loss 8.36 → 7.45；DPO loss 0.6931 → 0.0015、`margin` -0.0 → 9.74。三段各自写出 `ckpt.pt` 与 `train_log.jsonl`，DPO 的 `ref` 文件全程只读未写。

`train-dpo` 的交付配置（cwd 为仓库根，`$P=projects/tiny-llm-pipeline`）：

```bash
./start.sh run tiny-llm-pipeline train-dpo \
  --data $P/artifacts/data \
  --prefs $P/artifacts/prefs/prefs.jsonl \
  --ref $P/artifacts/sft-30m/ckpt.pt \
  --out $P/artifacts/dpo-30m \
  --beta 0.01 --epochs 4 --lr 3e-5
```

产物 `artifacts/dpo-30m/`：`ckpt.pt`（104 步）、`train_log.jsonl`、`samples_sft_vs_dpo.txt`（同一批未见过的 prompt 上 SFT vs DPO，`repetition_penalty ∈ {1.0, 1.15, 1.3, 1.5}` 各一份，供比较）。`--ref` 全程只读未写。DPO 只写 `ckpt.pt`，不像 SFT 那样有 `monitor.png` 与 `best.pt`。

`--lr` 默认 `config.DPO_LR = 1e-5`，交付用的是 **3e-5**；`--beta` 与 `--epochs` 也都要显式给——默认的 `beta=0.1` 落在「通顺但没风格」的欠训点（Δ=104）。

`--max-steps` 是**绝对值**：`--resume` 时接着 ckpt 里的 `step` 数；fresh stage 从 0 起（base ckpt 的 `step` 属上一段，不继承，见 `test_step_counter_starts_fresh_from_base`）。`train-sft --limit` 控制读多少行。

`sft_t2t_mini` 里前后相邻的问答没有上下文关系。`prepare_sft` 先把每条记录拆成独立的一问一答，再做留出集、DPO 提示和训练集的切分。

`train-sft` 的存盘和预训练同一套：默认每 **500** 步写 `ckpt.pt`（优化器、累计 token、行游标）并记下 holdout 的 response-token ppl；每 **10** 步覆盖 `monitor.png`；`--val-every` 默认 **1000**，从 holdout 里按该步抽 5 条贪心作答，不重复固定的前 5 条。留出困惑度最低的那次另存 `best.pt`，返回的就是它，续跑时从日志里恢复已有最优值，后来更差的步盖不掉。第一次 Ctrl-C 在当前步结束后保存；`--resume <ckpt>` 从该步的行游标接着训，并截掉日志里步数更大的行。

`--lr` 默认 **2e-4**（`config.SFT_LR`），低于预训练的 `1e-3`：第一次用 `1e-3` 时留出困惑度在第 4,500 步见底后回升到 36.6，见「指标与基线」。

**注意 `--max-steps` 会决定余弦视野**：`--resume` 时提高它会把学习率重新抬高（5000 → 7000 时 `2.0e-5` 跳回 `5.4e-5`，困惑度 6.34 → 8.92）。要延长训练得保持原视野或显式给出更长的计划，别只是调大 `--max-steps`。`train-dpo` 同理：它的视野由 `--epochs × 每 epoch 步数` 或 `--max-steps` 决定，这就是上面扫描里「同样 beta 但不同 epochs 结果差很多」的直接原因。

`eval/results.jsonl` 尚未接入。`gen` / `chat` / `compare` 尚未落地；落地时 `gen` 必须暴露 `repetition_penalty`（默认约 1.3），否则交付的 DPO 权重会复读。

`eval/sample.py` 与 `eval/serve.py` 的实跑口径（2026-10-07，`checkpoints/`：pretrain 34500 / sft 5000 / dpo 104，mps）：`sample.py` 重构到共用的 `generate.py` 前后，同一参数、同一种子产出的 HTML **逐字节相同**。`serve.py` 上 `你好，介绍一下你自己。` 在贪心 + `rep_pen=1.5` 下三段都有文本（chars 116 / 75 / 74，`elapsed_ms` 1839），DPO 栏 `tempo=1`——前摇只出现一次、没有原地复读；同一问在 `temperature=0.8` 下 DPO 栏出现 U+FFFD，复现了「采样只掉连贯性、不加内容」的既有结论，所以页面的默认仍是贪心。

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
| `prefs.jsonl` | **否** | `a1f7af44084af10a` | DeepSeek API 生成，temperature 0.8，无 seed 无缓存；215/240 的产出取决于接口行为。当前 **822** 对，首轮 404 对的哈希是 `424e5826e641cc19`（交付 `dpo-30m` 用的就是那 404 对）|
| `doubao-style-guide.md` | **否**（源文档） | `0c8f1c15893af605` | 手写纲要，不是任何代码的输出 |
| `pretrain/ckpt.pt` | 同类可复现，**非位级** | `4fa36c13e19ceaf4` | 同 seed/数据/代码可重训，但 MPS 与 CUDA 浮点路径不同 |

`sft-30m/ckpt.pt` 与 `dpo-30m/ckpt.pt` 同理：同 seed / 数据 / 代码可重训，但 MPS 与 CUDA 浮点路径不同，不是位级。

`prepare_pretrain` 的切分是 `val_docs, train_docs = kept[:n_val], kept[n_val:]`：**`train.bin` 从第 `n_val` 篇文档开始，不是第 0 篇**。做前缀比对要拿 `val.bin`（从第 0 篇起）。

**跨机搬运**：`artifacts/` 整体被根 `.gitignore` 第 35 行 `projects/*/artifacts/` 忽略，新机器 `git clone` **拿不到任何产物**。搬到云端时：

- **必须搬** `artifacts/prefs/`（`prefs.jsonl` + `doubao-style-guide.md`，共约 730 KB）——代码生成不出 `prefs.jsonl`（要花钱调 API 且不位级复现），而 `doubao-style-guide.md` 是 `synth-pref --guide` 的默认值，缺了默认命令直接失败。
- `artifacts/data/` 4.2 GB 选择重生成或一起搬。重生成 = 重下 2.8 GB 原始语料 + 重跑 `prepare`（本机约 10 min），用上表哈希校验。
- `ckpt.pt` 不必搬，重训即可，且它本来就不是位级可复现的。

**依赖钉死的缺口**：`tokenizers` 与 `numpy` **没有**写进 `requirements.txt`，只由 `transformers==5.18.0` / `torch==2.13.0` 传递带入（当前 `tokenizers 0.23.2`、`numpy 2.5.3`）。`tokenizer.json` 的位级可复现依赖 BPE 训练器实现，换 tokenizers 版本可能改变 merges，连带 `train.bin` 与三份切分一起漂移——**换机后先重训 tokenizer 比对哈希，对齐再往下走**。另外 `requirements.txt` 是全仓库共用的，`./start.sh setup` 会把其他 6 个项目（含 `colpali-engine` / `streamlit` / `pytesseract` / `pymupdf`）连 `-e` 安装一起拖进来；云端建议按需最小安装。Linux GPU 机器还需自己确认 torch 拿到的是 CUDA wheel（`start.sh` 只在 Windows 走 `--torch-backend=cu126`）。

## 风险

见 design 第 14 节。模型冒烟与 200 step 预训练短跑已通过。预训练 token 预算已钉为 `PRETRAIN_TOKENS`；SFT 的留出困惑度与 DPO 的 reward margin 已实测（见「指标与基线」）。

**DPO 这条交付的主要风险**：权重学到的是豆包体的表面特征（前摇、宝子/本豆/～、共情、追问），不是「按题作答」——`repetition_penalty` 调到 1.3 以上时复读消失，但内容开始游离、编造书名。29M 的容量加 404 对离线偏好对，能到这一步但到不了可用。要往前推得动数据侧（chosen 开头去模板化，甚至让 SFT 先见过豆包体），不是继续加 DPO 压力。
