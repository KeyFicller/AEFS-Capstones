# tiny-llm-pipeline

从零训练一个中文小模型，走通预训练、SFT、DPO。架构与指标见 `PROJECT.md`。

## 跑起来

```bash
./start.sh setup                                          # 共享 .venv + 依赖，只需一次
./start.sh test projects/tiny-llm-pipeline/tests          # 单测

P=projects/tiny-llm-pipeline
./start.sh run tiny-llm-pipeline prepare --out $P/artifacts/data   # 拉语料 → 训 tokenizer → 打包 bins 与切分
./start.sh run tiny-llm-pipeline synth-pref --n 400 \              # 合成 DPO 偏好对（追加式，中断可续跑）
    --prompts $P/artifacts/data/dpo_prompts.jsonl \
    --out $P/artifacts/prefs/prefs.jsonl --data $P/artifacts/data
./start.sh run tiny-llm-pipeline train-pretrain --data $P/artifacts/data \
    --out $P/artifacts/pretrain --max-steps 200                    # 预训练接龙
./start.sh run tiny-llm-pipeline train-sft --data $P/artifacts/data \
    --base $P/artifacts/pretrain/ckpt.pt --out $P/artifacts/sft --max-steps 200
./start.sh run tiny-llm-pipeline train-dpo --data $P/artifacts/data \
    --prefs $P/artifacts/prefs/prefs.jsonl --ref $P/artifacts/sft/ckpt.pt \
    --out $P/artifacts/dpo --max-steps 200                         # ref 冻结只读
```

`--max-steps` 是每个 stage 自己的步数；`--resume <ckpt>` 才接着原步数继续。想先验证整条链能跑，把三处的 `--max-steps` 都设成 `3`，并给 `train-sft` 加 `--limit 32`（`sft_train.jsonl` 有 119 万行，逐行流式读）。

`./start.sh run` 的 cwd 是仓库根，所以路径写仓库根相对；`tiny-llm-pipeline` 是 console entry point。
