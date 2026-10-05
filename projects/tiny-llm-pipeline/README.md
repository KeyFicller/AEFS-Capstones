# tiny-llm-pipeline

从零训练一个中文小模型，走通预训练、SFT、DPO。架构与指标见 `PROJECT.md`。

## 跑起来

```bash
./start.sh setup                                          # 共享 .venv + 依赖，只需一次
./start.sh test projects/tiny-llm-pipeline/tests          # 单测
./start.sh run tiny-llm-pipeline prepare --out projects/tiny-llm-pipeline/artifacts/data  # 拉语料 → 训 tokenizer → 打包 bins 与切分
```
