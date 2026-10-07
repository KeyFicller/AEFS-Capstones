# tiny-llm-pipeline

从零训练一个约 29M 的中文小模型（MiniMind2-Small 配置：词表 6400、宽 512、8 层），走通预训练接龙 → SFT → DPO 对齐豆包体，并附一个三阶段对照页。

![流程图](graph.png)

三段数据流：`prepare` 拉语料并切出预训练 bin 与三份互不相交的 SFT 切分 → `synth-pref` 合成偏好对 → `train-pretrain` / `train-sft` / `train-dpo` 逐段训练，每段都从上一段的 `ckpt.pt` 出发（DPO 的 `ref` 是冻结的第二份拷贝）。

## 跑起来

```bash
./start.sh setup                                        # 共享 .venv + 依赖，只需一次
P=projects/tiny-llm-pipeline

./start.sh run tiny-llm-pipeline prepare --out $P/artifacts/data    # 拉语料 → 训 tokenizer → 打包 bins 与三份切分
./start.sh run tiny-llm-pipeline synth-pref --n 400 \               # 合成偏好对；追加式，中断可续跑
    --prompts $P/artifacts/data/dpo_prompts.jsonl \
    --out $P/artifacts/prefs/prefs.jsonl --data $P/artifacts/data
./start.sh run tiny-llm-pipeline train-pretrain --data $P/artifacts/data --out $P/checkpoints/pretrain
./start.sh run tiny-llm-pipeline train-sft --data $P/artifacts/data \
    --base $P/checkpoints/pretrain/ckpt.pt --out $P/checkpoints/sft --max-steps 5000
./start.sh run tiny-llm-pipeline train-dpo --data $P/artifacts/data \
    --prefs $P/artifacts/prefs/prefs.jsonl --ref $P/checkpoints/sft/ckpt.pt \
    --out $P/checkpoints/dpo --beta 0.01 --epochs 4 --lr 3e-5        # 交付配置，--ref 全程只读

./start.sh test projects/tiny-llm-pipeline                          # 单测
./start.sh run python projects/tiny-llm-pipeline/eval/sample.py --rep-pen 1.5 --html $P/eval/replies.html   # 离线抽样 HTML
./start.sh run python projects/tiny-llm-pipeline/eval/serve.py --open                                      # 三段对照页
./start.sh run python projects/tiny-llm-pipeline/checkpoints/sync_ckpt.py pull                                # 新机器：拉回三段权重
```

只验证接线：三处 `--max-steps` 设 `3`、`train-sft` 加 `--limit 32`（`sft_train.jsonl` 有 112 万行，逐行流式读）；`synth-pref` 需自备 `DEEPSEEK_API_KEY`。

三段实测回复见 `eval/replies.html`，三段权重公开在 [HF `KeyFicller/tiny-llm-pipeline-29m`](https://huggingface.co/KeyFicller/tiny-llm-pipeline-29m)。生成必须开 `repetition_penalty`、`--max-steps` 决定余弦视野、`checkpoints/` 与 `artifacts/` 被 gitignore（`artifacts/prefs/` 需随机器搬运）——这些口径见 `PROJECT.md` 的「交付物」。
