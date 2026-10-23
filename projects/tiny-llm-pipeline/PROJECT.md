# tiny-llm-pipeline

- **项目**：tiny-llm-pipeline / 所属 Phase：Phase19 / Capstone 07
- **spec**：[Capstone 07 — 端到端微调流水线](https://aieng-zh.cn/lessons/19-capstone-projects/07-end-to-end-fine-tuning-pipeline/)
- **状态**：三段权重已交付并公开在 HF；`eval` 子命令是唯一报数来源；单测 `122 passed`。**未落地**：`gen` / `chat` / `compare` CLI、LoRA 与全量 FT 的 5000 步对照（本机未跑满，待云端 GPU）。

## 目标与范围

**目标**：从零训练一个约 29M 的中文小模型，走通预训练接龙 → SFT → DPO 对齐豆包体。

- **模型**：`ModelConfig`——词表 **6400**、宽度 512、8 层、8 个 query 头、2 个 KV 头；FFN 宽度取仓库公式 `ceil(512 * pi / 64) * 64 = 1664`；绑定词嵌入后 **28,975,616** 参数。训练窗口 `max_seq_len = 768`。
- **交付读数**（40 条冻结 prompt、贪心、`rep_pen=1.5`、`max_new=128`，全部来自 `eval/results.jsonl`）：

| 指标 | pretrain 34500 | sft 5000 | dpo 100 |
| --- | --- | --- | --- |
| 风格分 `doubao_score` | 0.093 | 1.059 | **2.326** |
| 排比开场（前 30 字含 ≥2 个「最」） | 0/40 | 0/40 | **18/40** |
| 跨题 4-gram 多样性 | 0.848 | 0.840 | 0.805 |
| 平均字数 | 192.3 | 116.2 | 72.1 |
| 留出集 response ppl | 15.1 | 35.9 | 41.0 |

- DPO 相对 SFT 的配对差 **+1.267**，bootstrap 2000 次 95% CI **[0.549, 1.956]**，DPO 胜 **77.5%** / 平 10% / SFT 胜 12.5%。
- 目标是让模型习得「最…最…」排比的豆包体开场：预训练与 SFT 都是 0/40，DPO 后 18/40。风格分上涨的同时**必须并列报退化护栏**（`repeat_4`、跨题多样性、平均字数、留出 ppl），否则「更短更密集」会被误读成对齐。DPO 的 ppl 高于 SFT 是预期内的——它专精一个窄语域。

## 架构

均在 `src/tiny_llm_pipeline/` 下：

| 路径 | 职责 |
| --- | --- |
| `config.py` | `ModelConfig` / `TrainConfig` / `PRETRAIN_TOKENS`，所有可调项 |
| `model.py` | `TinyLM`（tied embedding）；普通 ckpt 与 adapter ckpt 的读写 |
| `lora.py` | `LoRAConfig` / `LoRALinear` / `inject_lora` / `merge_lora` / adapter 存取 |
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
| `cli.py` | `prepare` / `synth-pref` / `train-pretrain` / `train-sft` / `train-dpo` / `eval` / `flow` / `monitor` |
| `evaluate.py` | 冻结题集上的确定性评测 + `results.jsonl`（run / item / summary） |

**图产物**：`graph.png` 由 `flow` 子命令生成，拓扑改了重跑。README 只放这一张图。

## 技术栈

Python 3.12.13，共享 venv。`torch`（fp32；设备按 CUDA → MPS → CPU 的顺序选取）、`tokenizers`（钉 `==0.23.2`）、`numpy`、`langchain-deepseek`、`typer`、`rich`、`matplotlib`、`pytest`。不新增依赖。

## 交付物

**权重**：三段公开在 [HF `KeyFicller/tiny-llm-pipeline-29m`](https://huggingface.co/KeyFicller/tiny-llm-pipeline-29m)，布局与本项目一致，`Bundle.load(<snapshot>)` 可直接加载、无需改代码。**已剥离 `optimizer`**（因此不能 `--resume` 续训），权重与训练产物逐张量相同。推/拉用 `checkpoints/sync_ckpt.py push` / `pull`：`pull` 会**拒绝覆盖含 `optimizer` 的可续训完整 ckpt**（Hub 上是瘦身版，覆盖即永久失去 `--resume`），要覆盖得显式 `--force` 或改写到 `--out`；`push` 只上传剥离后的权重，靠 `checkpoints/.sync_manifest.json` 记源 ckpt 的 sha256，源没变就沿用已暂存字节（`torch.save` 字节不稳定，否则每次 push 都造一个内容等价的空提交）。`--stages` / `--dry-run` 可分别选段与空跑。**Hub 上的 `dpo/train_log.jsonl` 与 `dpo/ckpt.pt`（100 步）不同源**，读 DPO 曲线以 ckpt 的 `step` 为准。

**数据与 `artifacts/`**：`checkpoints/` 与 `projects/*/artifacts/*` 被根 `.gitignore` 忽略，新机器 `git clone` 拿不到。唯一强制入库的是 `artifacts/prefs/`——`prefs.jsonl` 靠调 API 生成、**不可位级复现**，必须随机器搬运；`doubao-style-guide.md` 是 `synth-pref --guide` 的默认值，`style-fewshot.md` 是 `style-sft` 的默认值，缺了默认命令直接失败。**权重与它的 tokenizer 必须一起搬运**：BPE 的 id 分配跨版本不稳定，重训 tokenizer 会换出一份 id 全排列不同的词表，已有 ckpt 的 embedding 行与新 id 无对应关系，等于从头再训。`prepare` 见到 `--out/tokenizer.json` 就复用、不再静默重训（`--retrain-tokenizer` 强制重训）。

- **预训练数据**：`prepare` 从 HF 数据集仓库拉 `pretrain_hq.jsonl` / `sft_mini_512.jsonl`（仓库 id 见 `data.py` 的默认参数；走 HF 缓存，重跑只付一次下载），在 200MB 采样上训词表 6400 的 byte-level BPE，再打包 `train.bin` / `val.bin`。**必须钉 `DATASET_REVISION`**：该仓库的 `main` 已把这两个文件换掉，不钉指定 commit 会在新机器上 404。派生文件在同 `tokenizers` 版本下位级可复现；切分另依赖 `--seed`（默认 42）。
- **SFT 数据**：源文件里前后相邻的问答**没有上下文关系**，`prepare_sft` 先把每条记录拆成独立的一问一答，按 `--seed` 打乱后再切出**互不相交**的三份——`sft_train.jsonl` / `sft_holdout.jsonl` / `dpo_prompts.jsonl`。holdout 不做文件前缀切分（源文件顺序与长度/主题相关，前缀有偏）。
- **偏好对**：`synth-pref` 读 `dpo_prompts.jsonl`，写 `{prompt, chosen, rejected, model, created_at}`，追加式、默认续跑（已在输出里的 prompt 跳过，中断重跑不重复计费）。**两侧是同一答案的两个语域**：先按豆包体生成 `chosen`，再让它把 `chosen` **改写成中性口吻**（保留事实、结论、结构，长度相当）。独立采样会让一对样本同时改变内容、长度、语气，DPO 于是主要在学「答短点」。**丢弃不截断**：拒答、两侧相同、`prompt + 回答` 超过 `max_seq_len` 的对一律丢掉并计数，同时写 `<stem>.skipped.jsonl` 旁挂文件，`reason` 分 `refusal:chosen` / `refusal:rejected` / `empty` / `identical` / `over_limit`。拒答只判开头：`无法` / `不能` / `抱歉` 都是普通中文词，扫全文会把正常回答误杀。交付数据 `prefs.jsonl` 985 对 / 15 条真拒答，覆盖 1000 条 prompt。
- **开场词按 prompt 哈希分配**：风格纲要里的例句会被模型照抄，`chosen_system` 因此带一条 `_NO_COPY`（例句只是示范、每条换一种开场、不要每次从「宝子」起头），另按 prompt 的 sha256 从严选 24 词池里取 3 个「最 X」写进 system prompt——2024 种组合，跨重启与分片一致。只约束词汇、不约束句式：按角度轮转会让每条都从同一个骨架起头。`synth_one` 每轮打印两侧 token 数差中位数。
- **风格预热（`style-sft`）**：DPO 教不会模型一个语料里没出现过的语域。`style-sft` 把 `prefs.jsonl` 的 `chosen` 转成 `--data` 目录（复制 tokenizer，**丢掉照抄风格纲要的对**），过滤后得 682 条（train 613 / holdout 69）。

**训练（交付配置）**

- 预训练 34500 步（val ppl **11.8**，423M token）→ SFT 5000 步（留出回答 ppl **6.34**，微调前基线 **38.15**）→ DPO **100 步**（`--beta 0.1 --epochs 1 --lr 1e-5 --max-steps 100`，`--ref` 全程只读）。
- **生成时必须开 `repetition_penalty`（1.3–1.5）**：它是交付的一部分，不是可选润色。
- **存盘**：每 **500** 步写 `ckpt.pt`（优化器 / 累计 token / 行游标 / `horizon`）。`ckpt.pt` 每步覆盖同一路径，SFT 另有 `best.pt`（留出困惑度最低的那次，也是返回值）兜住末步覆盖。一次 run 拥有自己的日志：fresh run 覆盖日志并删掉上一次的 `best.pt`，`--resume` 只截掉步数大于 ckpt 的行。
- **`--max-steps` 是绝对值**，fresh stage 从 0 起（base ckpt 的 `step` 属上一段，不继承）。**余弦视野属于 run**：`horizon` 记在第一个 ckpt 里，`--resume` 若按新的 `--epochs` / `--max-steps` 算出不同视野就直接报错，报错里给出该传的值。
- **SFT batch 走 `seed` 派生的行置换**（不再是文件顺序），`--resume` 用同一 `(行数, seed)` 重建置换；预训练 `--resume` 从数据游标与累计 token 接着训，不重放看过的窗口。
- **DPO 按 prompt 的 sha256 留出 10%**（`--holdout-frac`），只在其余对上训练；每个存盘点在留出对上写 `holdout_gap` / `holdout_margin` / `holdout_win_rate`，并附逐 token 口径 `holdout_gap_norm` / `holdout_win_rate_norm`（求和口径偏爱短的一侧，两口径分叉即长度差在替风格干活）。
- **训练中止护栏**（`train/dpo.py` 顶部 `GUARD_*`）：每 10 步复查留出集下一词准确率，跌破参考的 **75%**，或 rejected 侧下压 **>20 nats 且 >2 倍** chosen 涨幅，即抛错中止。`holdout_gap` 只说明两侧在拉开，不说明模型是否被训坏。
- **`weight_decay = 0.1` 只作用于投影矩阵**：RMSNorm 的 gain 与 tied embedding 不进衰减组。

**评测（`eval` 子命令，唯一允许的报数来源）**

- 命令 `tiny-llm-pipeline eval --stages pretrain,sft,dpo --prefix "用豆包体回答：" --rep-pen 1.5 --max-new 128 [--bundle checkpoints] [--prompts eval/prompts.jsonl] [--out eval/results.jsonl] [--data artifacts/data]`；默认按**项目目录**解析，从仓库根或项目目录跑都一样。
- 题集 `eval/prompts.jsonl` 40 条（入库、冻结、不临场挑），与偏好对的 prompt **零交集**。解码固定贪心（`temperature=0`）；`--prefix` 是题面的一部分（风格切换靠它，`prompts_sha256` 覆盖前缀后的文本），`rep_pen` / `max_new` 同样是交付口径，改动即改题面，run 头会自动记下。
- 指标：`doubao_score`（25 词加权词表密度，按每 100 字计分）、`distinct_1` / `distinct_2` / `repeat_4`（单条内 4-gram 重复占比）、`longest_run`（最长同字连跑）。汇总另有 **`cross_prompt_distinct_4`**：把同一段全部回答拼起来算 4-gram 多样性——单条 `repeat_4` 抓不到「每条回答同一个开场白」，这一项才抓得到。
- 配对与区间：`doubao_delta_dpo_minus_sft` 给 `mean` / `ci95`（bootstrap 2000 次重采样，种子固定）/ `dpo_win_rate` / `tie_rate` / `sft_win_rate`。空输出照记（`chars=0` 仍是一行），不跳过。
- 行结构：`kind="run"`（commit / device / tokenizer hash / steps / 题集 sha256 / 解码参数 / 依赖版本）→ 逐项 `kind="item"` → `kind="summary"`；**每行写完即 flush**，中途失败已得结果不回滚；每次 run 覆盖同名文件而非追加。
- 本机缺 `artifacts/data`，重跑 `eval` 不会产出 `ppl` 段（各段在 `sft_holdout.jsonl` 上的 response-token 困惑度）；已入库的 `results.jsonl` 里那段是数据还在时跑出来的。
- **未做**：LLM 配对裁判（`eval/judge.py`）。`doubao_score` 是代理指标，报读数时必须并列退化护栏。

**推理与查看**：`eval/sample.py` 离线抽样 + 自包含 HTML，种子由 `(seed, stage, rep_pen, prompt, index)` 派生，同一批可复现、与执行顺序无关；`eval/serve.py` 本地对照页，三段并排，只绑 `127.0.0.1`、无鉴权、**不落盘**，默认 `rep_pen=1.5`、`temperature=0`（贪心）、`max_new=64`。两者共用 `generate.py`：同一参数、同一种子产出的 HTML 逐字节相同。采样（`temperature=0.8`）只掉连贯性、不加内容，故页面默认仍是贪心。

**SFT 的 LoRA 路径（已落地；对照实验待云端）**

- 入口 `train-sft --lora-rank 8 --lora-alpha 16`；**`--lora-rank 0` 是默认值，等于全量微调，行为一字未变**。只打 attention 的 `q_proj` / `k_proj` / `v_proj` / `o_proj`，`b` 零初始化（刚注入时 forward 与 base 逐位相等）。可训参数 **212,992 / 28,975,616 = 0.735%**，其余 `requires_grad=False`。
- 产物是 adapter ckpt（`cfg` / `lora` / `base` / `adapter` / `step` / `seed` / `tokenizer_hash`，**没有 `model` 键**）；`load_ckpt` 认 adapter 且**加载即 merge**，返回与全量 ckpt 同形的 `TinyLM`，故 `eval_sft_ppl`、`Bundle.load`、DPO 的 `--ref` 零改动即可读。`--resume` 走 `load_lora_ckpt`，不 merge。base 缺失 / base 又是 adapter / tokenizer hash 或 `cfg` 不符一律硬报错，不降级。
- **未运行**：5000 步对照。参考组同参数换 `--lr 1e-3`，读数口径为同一份 `sft_holdout.jsonl` 上的 `eval_sft_ppl`，与全量 FT 的 **6.34** 比；LoRA 更差就如实写。

**未落地**：`gen` / `chat` / `compare`。落地时 **`gen` 必须暴露 `repetition_penalty`（默认约 1.3）**，否则交付的 DPO 权重会复读。
