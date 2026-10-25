# 检索延迟优化记录

范围：经典链路 `bm25 + maxsim + rerank`，即 `--mode hybrid-maxsim` 再叠一层 VLM rerank。
其他臂（`ocr` / `abstract` / `pool` / `lexical-kw`）不参与优化，也不在本文回写数字。

## 怎么量

```bash
cd /Users/keyficller/Documents/AEFS-Capstones
export DEEPSEEK_API_KEY=$(rg '^DEEPSEEK_API_KEY=' local.env | cut -d= -f2-)

# 小语料（artifacts/，86 页）
./start.sh run python projects/multimodal-doc-qa/latency_optimize/bench_retrieval.py \
    --reps 8 --limit 5 --rerank --json projects/multimodal-doc-qa/latency_optimize/baseline.json

# 大语料（artifacts-mmdocir/，2600 页）：换 MDQ_ARTIFACTS_DIR，问题文件跟着目录走
MDQ_ARTIFACTS_DIR=projects/multimodal-doc-qa/artifacts-mmdocir \
./start.sh run python projects/multimodal-doc-qa/latency_optimize/bench_retrieval.py \
    --reps 3 --limit 3 --rerank --json projects/multimodal-doc-qa/latency_optimize/baseline-mmdocir.json
```

- `start.sh` 不加载 `local.env`，跑 `--rerank` 要自己 export key；不加 `--rerank` 不需要 key。
- 默认只测 `maxsim,lexical,hybrid-maxsim`；`--rerank` 每 query 只调一次网络，所以不受 `--reps` 影响。
- `bench_retrieval.py` 在 GPU 阶段后同步 MPS，否则 forward 是异步发射，成本会记到下一个阶段上。
- 额外量两行纯排名成本（`index.search` / `score loop`）：query 向量先算好不计内部，用来把「编码」和「扫库」分开。
- 读数字看 **min** 做阶段归因（min 之间能对上：encode + search + 后处理 ≈ invoke）；median 会被同进程里的内存分配和热漂移带偏 ±30%。

## 测到哪一层

**每行 = 一次检索 hop，不是一次 agentic 问答。** bench 直接调 `retriever.invoke(query)`，不经过 `graph.py`，所以不含：

- 入口意图分类（`classifier`）
- 查询改写（`augment`）
- 多轮：`assess` 生成 followup → 再检索，最多 `max_rounds=5` 轮，每轮每个 query 一次 hop
- 覆盖校验（`verify`）和作答（`synth`）

`--rerank` 那行对应 `retrieve_node` 里紧跟 `invoke` 的那次调用，所以在真实一轮 hop 里它和检索是串行的、成本要相加，且**每轮每个 query 都付一次**。一次完整问答的真实延迟 ≈ hop 数 ×（检索 + rerank）+ 若干次 LLM 调用，用这里的单 hop 数字乘上去估，别当成端到端结论。

## corpus 规模

延迟只在这个规模下成立。数字来自各目录的 `manifest.json`，patch 数从索引里实际数的。

| 项 | `artifacts/` | `artifacts-mmdocir/` |
| --- | --- | --- |
| 文档 | 4 篇 PDF（15 / 24 / 20 / 27 页） | 37 篇 PDF |
| 页 | 86 | 2,600 |
| MaxSim patch | 80,515（中位 875 / 页） | 2,768,942（中位 1,139 / 页） |
| BM25 chunk | 470（≤ 400 字符） | 19,399 |
| 渲染 | 180 dpi，长边 2048 | 同左 |
| multivector 索引 | 41.2 MB（479 KB / 页） | 1.42 GB（545 KB / 页） |
| BM25 语料索引 | 0.9 MB | 37.4 MB |
| 评测问题 | 19 条 | 312 条 |

规模决定瓶颈在哪，这两组数差 30 倍就是为了看这件事：

- **forward 是唯一不随语料涨的成本**：46ms，两组语料一样。
- **MaxSim 全量 CPU 扫描随页数线性涨**：86 页 5ms → 2600 页 106ms（30 倍语料、21 倍耗时，有固定开销摊薄）。按这个斜率 10k 页约 400ms、100k 页才到 4s。
- **BM25 是纯 Python 循环，同样线性**：470 chunk 0.9ms → 19,399 chunk 28ms，约 1.5µs/chunk。它比 MaxSim 更早出问题，因为 19k chunk 已经比 2.7M patch 更贵了。
- **rerank 与语料无关，只与送进去的页数有关**：`top_k=5` 张页图 + 一次 VLM 调用。86 页 1675ms、2600 页 2165ms —— 差值是网络抖动，不是语料。

## baseline A — `artifacts/`（86 页）

`device=mps`、`fp16`、`top_k=5`、5 条 query、86 页 / 80,515 patches。

一次性（每进程付一次）：

| 阶段 | 秒 |
| --- | --- |
| load MultiVectorEncoder (colSmol-500M) | 5.93 |
| load multivector index | 0.02 |
| load ocr index (BM25 语料) | 0.01 |
| build rerank model | 0.23 |

每 query（毫秒）：

| 阶段 | median | p90 | min |
| --- | --- | --- | --- |
| encode_query/processor | 1.00 | 1.22 | 0.60 |
| encode_query/forward | 45.65 | 47.76 | 35.44 |
| encode_query/transfer | 0.25 | 0.49 | 0.21 |
| encode_query/empty_cache | 1.41 | 1.77 | 1.21 |
| encode_query [total] | 48.64 | 50.59 | 42.87 |
| maxsim: invoke | 53.50 | 55.15 | 46.09 |
| lexical (bm25): invoke | 0.92 | 1.08 | 0.43 |
| hybrid-maxsim: invoke | 54.07 | 55.28 | 46.57 |
| rerank（1 次 LLM 调用） | 1675.24 | 2692.88 | 1195.95 |

这个规模下排名扫描（~5ms）被 forward 完全盖住，所以「检索层不是瓶颈」只在小语料成立。

## baseline B — `artifacts-mmdocir/`（2,600 页）

`device=mps`、`fp16`、`top_k=5`、3 条 query、2,600 页 / 2.77M patches。

一次性：

| 阶段 | 秒 |
| --- | --- |
| load MultiVectorEncoder | 6.00 |
| load multivector index (1.42 GB) | 0.49 – 0.95 |
| load ocr index (37 MB) | 0.12 – 0.20 |
| build maxsim | 0.42 – 0.46 |
| build lexical（19,399 chunk 切词 + 建表） | 0.38 – 0.41 |
| build hybrid-maxsim | 1.05 – 1.08 |

每 query（毫秒）：

| 阶段 | median | p90 | min |
| --- | --- | --- | --- |
| encode_query/processor | 1.04 | 1.32 | 0.65 |
| encode_query/forward | 41.98 | 47.05 | 35.47 |
| encode_query/transfer | 0.26 | 0.53 | 0.22 |
| encode_query [total] | 44.66 | 49.06 | 43.54 |
| maxsim: index.search | 105.95 | 118.48 | 104.14 |
| maxsim: invoke | 175.91 | 193.10 | 155.41 |
| lexical: score loop | 28.40 | 33.32 | 23.82 |
| lexical: invoke | 46.24 | 56.82 | 38.04 |
| hybrid-maxsim/dense: index.search | 110.09 | 122.71 | 107.14 |
| hybrid-maxsim: invoke | 210.75 | 240.37 | 202.32 |
| rerank（1 次 LLM 调用） | 2164.62 | 2382.92 | 1982.22 |

## 结论

按 2,600 页这组口径（大语料才代表上线形态）：

1. **rerank 仍是最大的一块，但优势从 31 倍缩到 10 倍**：2165ms vs 检索 211ms。它随 top_k 涨、不随语料涨。
2. **检索内部已经换主角了**：forward 44ms + MaxSim 扫描 106ms + BM25 ~28ms + RRF/截断 ~8ms。**扫描反超了模型推理**，两者合计占 71%。小语料时这个结论是反的 —— 优化目标要跟着语料规模走。
3. **BM25 是最便宜的加速点**：纯 Python 循环打 19,399 个 chunk 花 28ms，一点没优化过（`Counter` 查表 + math.log，无倒排表、无候选裁剪）。加个倒排索引就能砍掉一个数量级；MaxSim 那 106ms 则要动检索结构（聚类/量化/IvFPQ）才有得降。
4. **一次性成本仍主要是模型**：6.0s 加载视觉编码器；1.42GB 索引只要 0.5–0.95s（页缓存命中），`build lexical` 的 0.4s 也基本是切词 19k chunk。
5. **两处「假账」要记住**：之前以为的 `.cpu()` 传 22ms 是 forward 异步发射记错了账（同步后 0.25ms）；`encode_query` 末尾的 `torch.mps.empty_cache()` 则是真白付 1.4–1.6ms。

## 迭代记录

每条一行：假设 → 改动 → before/after → 保留还是回滚。检索 p50 取 `hybrid-maxsim: invoke`，口径见上两节。

| # | 日期 | 假设 | 改动 | 检索 p50 | rerank p50 | 结论 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 10-25 | — | baseline（86 页） | 54.07 | 1675.24 | — |
| 1 | 10-25 | 换大语料后瓶颈会变 | 加 `MDQ_ARTIFACTS_DIR` 支持 + `index.search` / `score loop` 拆行；量 2600 页 | 210.75 | 2164.62 | 扫描反超推理；BM25 是最便宜的加速点 |
