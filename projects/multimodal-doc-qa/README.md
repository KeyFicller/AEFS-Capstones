# multimodal-doc-qa

把文档页当图像做后期交互检索，再叠一层有界 agent 循环带引用作答；同一张图结构只换 retriever，对照 vision-first（ColPali 多向量 / MaxSim）与 OCR-first。

![REPL 实跑](example.png)

循环骨架：`plan` → `retrieve` → `assess` ⟲ → `synthesize` → `verify` ⟲，页面池空了或轮数用尽走 `recover`。实线往下走，虚线是打回或收口。

![图结构](graph.png)

## 跑起来

```bash
./start.sh setup                          # 共享 .venv + 依赖，只需一次
brew install tesseract                    # 扫描页的 OCR 臂需要它
./start.sh run doc-qa ingest --corpus <pdf_dir>   # 已有 PDF / 图片 / txt / md
./start.sh run doc-qa                     # 多轮 REPL；--mode vision|pool|ocr|summary，默认 vision
./start.sh run pytest projects/multimodal-doc-qa/tests -m "not slow"   # 单测（slow 会加载真实模型）
```

评测（两条手臂分开落盘，查看器才能并排；题目文件由调用方提供）：

```bash
./start.sh run doc-qa eval --mode vision --questions <questions.json> --out projects/multimodal-doc-qa/eval/vision.jsonl
./start.sh run doc-qa eval --mode ocr    --questions <questions.json> --out projects/multimodal-doc-qa/eval/ocr.jsonl
```

引用框（不需要标准答案；`ask` / REPL 会把每一轮写进 `artifacts/turns.jsonl`）：

```bash
./start.sh run streamlit run projects/multimodal-doc-qa/src/multimodal_doc_qa/ui/viewer.py
```

侧栏的结果文件留空时读这份 `turns.jsonl`。评测产出的 `results.jsonl` 仍可填进去并排；只有目录里有 `questions.json` 时才画蓝色标准答案框。

`ask` / REPL / `eval` 需要 `DEEPSEEK_API_KEY`（CLI 自读仓库根 `local.env`）。换编码器必须重新 `ingest`。指标口径与预算见 `PROJECT.md`。
