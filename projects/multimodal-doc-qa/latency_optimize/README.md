# 检索延迟优化记录

范围：经典链路 `bm25 + maxsim + rerank`，即 `--mode hybrid-maxsim` 再叠一层 VLM rerank。
其他臂（`ocr` / `abstract` / `pool` / `lexical-kw`）不参与优化，也不在本文回写数字。

## 怎么量

```bash
cd /Users/keyficller/Documents/AEFS-Capstones
export DEEPSEEK_API_KEY=$(rg '^DEEPSEEK_API_KEY=' local.env | cut -d= -f2-)

MDQ_ARTIFACTS_DIR=projects/multimodal-doc-qa/artifacts-mmdocir \
./start.sh run python projects/multimodal-doc-qa/latency_optimize/bench_retrieval.py \
    --reps 3 --limit 3 --rerank --json projects/multimodal-doc-qa/latency_optimize/baseline-mmdocir.json
```

- `MDQ_ARTIFACTS_DIR` 一换，索引和问题文件（`questions.json`）都跟着目录走。
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

`--rerank` 那几行对应 `retrieve_node` 里紧跟 `invoke` 的那次调用，所以在真实一轮 hop 里它和检索是串行的、成本要相加，且**每轮每个 query 都付一次**。`baseline, one query at a time` / `batch<=4` 是同一件事的两种调度；产品只走 batch，baseline 只为对比保留。一次完整问答的真实延迟 ≈ hop 数 ×（检索 + rerank）+ 若干次 LLM 调用，用这里的单 hop 数字乘上去估，别当成端到端结论。

## corpus 规模

延迟只在这个规模下成立。数字来自 `artifacts-mmdocir/manifest.json`，patch 数从索引里实际数的。

| 项 | 值 |
| --- | --- |
| 文档 | 37 篇 PDF |
| 页 | 2,600 |
| MaxSim patch | 2,768,942（中位 1,139 / 页，128 维） |
| BM25 chunk | 19,399（每 chunk ≤ 400 字符） |
| 渲染 | 180 dpi，长边 2048 |
| multivector 索引 | 1.42 GB（545 KB / 页，落盘 float32） |
| BM25 语料索引 | 37.4 MB |
| 评测问题 | 312 条（bench 默认只取 3 条不同的） |

## baseline — `artifacts-mmdocir/`（2,600 页）

`device=mps`、`fp16`、`top_k=5`、3 条 query、2,600 页 / 2.77M patches。

一次性（每进程付一次）：

| 阶段 | 秒 |
| --- | --- |
| load MultiVectorEncoder (colSmol-500M) | 6.00 |
| load multivector index (1.42 GB) | 0.49 – 0.95 |
| load ocr index (37 MB) | 0.12 – 0.20 |
| build maxsim | 0.42 – 0.46 |
| build lexical（19,399 chunk 切词 + 建表） | 0.38 – 0.41 |
| build hybrid-maxsim | 1.05 – 1.08 |
| build rerank model | 0.26 |

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

1. **rerank 仍是最大的一块**：2165ms vs 检索 177ms。它随 top_k 涨、不随语料涨。
2. **检索内部的主角是 MaxSim 扫描**：MaxSim 106ms + 编码 query 45ms + BM25 4.0ms + RRF/截断 ~8ms。扫描占 60%，编码 25%，BM25 已降到 2%。
3. **BM25 已经不再是加速点**：倒排 + 上界剪枝后 40ms → 3.1ms。剩下能降的只有 MaxSim 那 106ms，而它要动检索结构（聚类/量化/IvFPQ）才有得降，且必须 eval 兜底。批量化反而更慢（一个 GEMM + segment max 要 263ms，小 matmul 才 105ms）。
4. **一次性成本仍主要是模型**：6.0s 加载视觉编码器；1.42GB 索引只要 0.5–0.9s（页缓存命中）；倒排表把 `build lexical` 从 0.39s 推到 0.48s、内存约 7MB。
5. **两处「假账」要记住**：之前以为的 `.cpu()` 传 22ms 是 forward 异步发射记错了账（同步后 0.25ms）；`encode_query` 末尾的 `torch.mps.empty_cache()` 则是真白付 1.4–1.6ms。

## 迭代记录

每条一行：假设 → 改动 → before/after → 保留还是回滚。检索 p50 取 `hybrid-maxsim: invoke`，rerank 取单次调用（一次 rerank 请求）的 median。

| # | 日期 | 假设 | 改动 | 检索 p50 | rerank p50 | 结论 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 10-25 | 大语料下瓶颈会变 | 加 `MDQ_ARTIFACTS_DIR` 支持 + `index.search` / `score loop` 拆行；量 2600 页 | 210.75 | 2164.62 | 扫描反超推理；BM25 是最便宜的加速点 |
| 1 | 10-25 | 多 query 的 rerank 可以并发 | listwise 契约不变；`rerank_pages` 改 async，`aio.run` + `Semaphore(4)` gather；串行作为基线留在 bench，产品只走 batch | 210.75（未变） | 单次调用中位 2164.62（单独）→ 5145（4 路 batch 内） | 16 query 墙钟 22690 ms。上行在 2 路就饱和，4 路墙钟最低；真正的杠杆是 12.52 MB 的 payload |
| 2 | 10-25 | BM25 可以把大多数 chunk 排除掉 | 启动时建倒排表；查询取 posting 并集，按上界降序打分，上界低于第 k 名即 break | 210.75 → **177.28**；lexical 40.04 → **3.10** | 未复测 | 13×。312 条评测问题的排名与原全扫逐位相同。`build lexical` 0.39 → 0.48s |

### 第 1 条的细节

- 默认 `augment=rewrite` 只产 1 条 query → **batch 收益 0**，必须 `MDQ_AUGMENT=multiquery` 或 `decompose` 才有得并发。
- 代价是线性的：3 条 query = 3 次 rerank 调用 = 成本 ×3、token ×3。
- bench 同一轮里交替跑串行与 batch、各 `RERANK_ROUNDS=3` 次取中位：跨进程取样会撞上服务端低谷，串行/并发分两次进程跑出来的对比不可信。
- 实现坑：不能用 `asyncio.run`。同一个 model 连做两次 `asyncio.run(ainvoke)` 第二次必报 `Event loop is closed`（连接池绑在首个 loop 上），多轮 retrieve 一定踩到。改成 `aio.py` 的常驻 loop。
- 用例 `test_rerank_runs_queries_concurrently_under_a_cap` 断言同时在飞峰值 > 1 且 ≤ `RERANK_CONCURRENCY`，证明并发有界且真实。

### 真正的杠杆是一个 rerank 请求 12.5 MB

`build_page_blocks` 把 5 张页图编成 **PNG data URL**（180 dpi、长边 2048）。实测单请求 payload：

| | 原始字节 | base64 后 |
| --- | --- | --- |
| 5 张页图 | 9.39 MB | **12.52 MB** |

改图片尺寸，延迟跟着字节走，说明瓶颈在上传：

| payload | 大小 | 单请求 |
| --- | --- | --- |
| 原样 PNG 2048 | 12.52 MB | 3031 ms |
| JPEG 长边 768 | 0.64 MB | 1022 ms |
| JPEG 长边 384 | 0.17 MB | 1055 ms |

12.52→0.64 MB 省下 2009 ms，即 11.88 MB / 2.0 s ≈ **5.9 MB/s ≈ 47 Mbps 上行**。再压到 0.17 MB 不再变快（1022 vs 1055），那 ~1 s 是模型的固定计算地板。

并发的上限也由这条水管决定。16 query × 5 页 = 80 次调用，每次 12.52 MB，原样 PNG：

| 并发 | 墙钟 | 单次调用中位 | 有效上行 | 相对 1 路 |
| --- | --- | --- | --- | --- |
| 1 | 34976 ms | 1943 ms | 5.73 MB/s | 1.00× |
| 2 | 23450 ms | 2530 ms | **8.54 MB/s** | 1.49× |
| 4 | **22690 ms** | 5145 ms | **8.83 MB/s** | 1.54× |
| 8 | 23527 ms | 10136 ms | 8.51 MB/s | 1.49× |
| 16 | 24690 ms | 17131 ms | 8.11 MB/s | 1.42× |

- 1→2 是唯一实打实的收益；2 之后上行锁死在 ~8.5 MB/s，墙钟 22.7–24.7s 持平。**并发不能让水管变粗。**
- 4 比 2 只快 3%，在噪声内。真正被并发上限决定的是单次延迟：2 路 2530 ms → 16 路 17131 ms，纯排队。
- 16 路墙钟反而变差，且配上 `max_ask_seconds` 更容易撞超时。所以常数取 4（墙钟最小值），不是 8。

换成 768px JPEG 后同样的 3 路并发：

| payload | 3 query 并发 wall |
| --- | --- |
| 原样 PNG 2048 | 9043 ms |
| JPEG 768 | **2340 ms** |

**3.9×**，不需要多花一分钱、多调一次 API。这条还没落地：`build_page_blocks` 仍走 PNG 2048。

质量前提：rerank 只是从 5 页里排序，看缩略图足够；需要读小字的是 answerer（synth），那条路径要保留原分辨率（它还要出 bbox 引证）。

### 第 2 条的细节

只建倒排表不够：7 个词的 query 取并集后候选还有 84%，`the` 一个词就命中 83.8% 的 chunk。所以必须叠加**上界剪枝** —— 一个词最多贡献 `idf × (k1+1)`，按这个上界从高到低打分，一旦某候选的上界低于当前第 k 名，它和它后面的全部跳过（融合臂的 `stage_k` 是 20，不是 5）。

| k | 全扫 | 剪枝 | 实际打分 |
| --- | --- | --- | --- |
| 5 | 21.99 ms | 2.96 ms | 48 – 547 |
| 20 | 22.25 ms | 4.38 ms | 161 – 1,608 |
| 40 | 22.25 ms | 4.71 ms | 415 – 1,709 |

剪枝是**精确**的：被剪掉的 chunk 数学上不可能进 top-k，312 条评测问题逐位复现原排名。代价是启动时多 95 ms 建表、约 7 MB 内存（37,062 词 / 848,241 条 posting），不落盘。

一个实现坑：维护 top-k 必须用 `heapq`。用 `sorted()` 每次重算第 k 名会让 k=20 变成 **75.82 ms**，比全扫还慢三倍。

一个边界：上界对「全是高频词」的 query 偏松。`what is the model` 仍能剪到 2.36 ms，但如果 query 的每个词都出现在大半语料里，剪枝会退化到接近全扫 —— 不会更慢，只是没收益。
