# multimodal-doc-qa

把文档页当图像做后期交互检索，再叠一层有界 agent 循环带引用作答；同一张图结构只换 retriever，对照 vision-first（ColPali 多向量 / MaxSim）与 OCR-first。

![REPL 实跑](example.png)

循环骨架：`intent` 先判这一轮是 `chat`（直接回、不检索）还是 `work` → `plan` → `retrieve` → `assess ⟲` → `synthesize` → `verify ⟲`。`assess` 判页面池不够、`verify` 判引用不支撑，都把循环打回 `retrieve`；池子空了或轮数用尽走 `recover` 收口。

![图结构](graph.png)

## 跑起来

```bash
./start.sh setup                          # 共享 .venv + 依赖，只需一次
brew install tesseract                    # 扫描页的 OCR 臂需要它
./start.sh run doc-qa ingest --corpus <pdf_dir>   # 已有 PDF / 图片 / txt / md；视觉索引落 multivector_index.pt（由 vision_index.pt 改名，升级时把旧文件改名即可）
./start.sh run doc-qa                     # 多轮 REPL；--mode maxsim|pool|ocr|abstract|lexical|hybrid-{ocr,abstract,pool,maxsim}，默认 maxsim
./start.sh run pytest projects/multimodal-doc-qa/tests -m "not slow"   # 单测（slow 会加载真实模型）
```

评测（两条手臂分开落盘，查看器才能并排；题目文件由调用方提供）：

```bash
./start.sh run doc-qa eval --mode maxsim --questions <questions.json> --out projects/multimodal-doc-qa/eval/maxsim.jsonl
./start.sh run doc-qa eval --mode ocr    --questions <questions.json> --out projects/multimodal-doc-qa/eval/ocr.jsonl
# 另有 lexical（手写 BM25）与四条 hybrid-<dense>（BM25 + 语义侧 RRF）臂，取值同上
./start.sh run streamlit run projects/multimodal-doc-qa/src/multimodal_doc_qa/ui/viewer.py   # 证据框叠加 + 并排
```

`eval` 除 nDCG / IoU 外还报检索侧指标：`recall@k`（裸检索器单趟找到多少 gold 页）与 `pool recall`（agent 循环累积池）。没有 `questions.json` 时，可用 `./start.sh run python -m multimodal_doc_qa.eval.prepare_gold --docs <a.pdf,b.pdf>` 从公开数据集 MMLongBench-Doc 取一个子集生成。

`ask` / REPL / `eval` 的凭据放仓库根 `local.env`；回答器默认 `deepseek:deepseek-flash`，`MDQ_ANSWERER_MODEL` 可换 provider / 模型。
