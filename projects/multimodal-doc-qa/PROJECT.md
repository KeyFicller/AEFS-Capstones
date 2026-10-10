# multimodal-doc-qa

- **项目**：multimodal-doc-qa（Agentic RAG）/ 所属 Phase：Phase19 / Capstone 04
- **spec**：[Capstone 04 — Multimodal Document QA](https://aieng-zh.cn/lessons/19-capstone-projects/04-multimodal-document-qa/)
- **状态**：`maxsim`（多向量后期交互）与 `ocr` 两条主线都已跑通——**同一张 agentic 图，只换 retriever**；`--mode` 现有十条检索臂；CLI 入口齐备（`ingest` / `ask` / `eval` / REPL）；单测 `309 passed`。`eval` 默认只跑检索器（`recall_at_k` / `nDCG@k`，不调 chat 模型），`--agentic` 才跑图并出 `pool_recall`（agent 循环累积池）与答案侧指标；gold 取自公开数据集 MMLongBench-Doc 的一个小子集，由 `python -m multimodal_doc_qa.eval.prepare_gold` 转换为 `questions.json`。**未实现**：答案准确率（`answer_containment`）、bytes/page、index p95。**已评测（检索臂消融）**：九条臂各跑纯检索 19 题（`eval/ablation-<arm>-19q.jsonl`，汇总 `eval/ablation-9arms-19q.html`）——`maxsim` 最高（recall@5 0.8947 / nDCG@5 0.7359），且已是这九条臂的上限（它只漏 2 题，此 2 题九臂全漏）；`lexical` 0.7368 / `abstract` 0.7105 / `pool` 0.6579 / `ocr` 0.6053；四条 `hybrid-*` 的 RRF 只救弱臂（`ocr` +0.18、`pool` +0.13），反拖低最强臂（`maxsim` −0.05）。`lexical-kw`（先让 LLM 抽词、再喂同一份 BM25）**不及 `lexical`**：recall@5 0.6842（−0.0526，一题从命中变全漏）、nDCG@5 0.6385（+0.0037，恰被另一题的排序改善抵消），延迟 p50 0.74s 对 `lexical` 的 0.002s——抽词会把问题里的字面线索（如格式要求 `["a","b"]`）改写成泛化短语，丢掉 BM25 赖以命中的判别词。**必须记住**：索引不记录写它的编码器，换 checkpoint 后必须重新 `ingest`。

## 目标与范围

**目标**：端到端做出一个 **Agentic RAG** 文档问答系统——把 PDF 页面当图像，用多向量**后期交互（late interaction）**检索取页；其上叠一层**有界 agent 循环**（`plan → retrieve → assess ⟲ → synthesize → verify ⟲`），由托管 VLM 带引用作答，在查看器里把证据区域叠回原页；并配一条 **OCR-first 文本流水线**跑同一批文档做基线。

agency 只改变「取哪些页」，不改变页面表示——检索与证据分项不受影响，多页推理受益。

**范围**

- **文档**：已有 PDF（以及图片 / `txt` / `md`）。`ingest` 读一个目录，**不生成语料、不写 ground truth**；题目与 gold 由调用方的 `questions.json` 给出，没有它 `eval` 不跑。
- **渲染**：PyMuPDF → 页面 PNG（180 DPI，长边归一化，保持原页长宽比）。
- **编码**：默认 `vidore/colSmol-500M`（MPS / float16）出多向量；4B 的 `vidore/colqwen2.5-v0.2` 经 `MDQ_EMBEDDER_MODEL` 换上，加载失败**直接报错，没有第二套 checkpoint**。
- **编码预算**：处理器 `max_pixels = 602112`（= 768 × 28²）。超过约 0.6 MP 的像素**在进编码器前即被丢弃**，故渲染 DPI 由 OCR 基线 / 查看器需求决定，而非检索质量。
- **索引 / 检索**：torch 张量精确 **MaxSim**（逐查询 token 取每页 patch 最大点积再求和）；`top_k` 默认 5（`MDQ_TOP_K`）；top-k 之后再过分数线，低于本轮最高分 `MDQ_MIN_SCORE_RATIO`（默认 `0.5`）的页不进结果。
- **合成**：回答器读**页面池**图 + 问题，`with_structured_output(Answer)` 直接产出带 `(doc_id, page, bbox?)` 引用的答案——**引用与 bbox 在合成时即结构化落地，无独立抽取步骤**。模型不给 bbox 即为页级引用。
- **OCR-first 基线**：PyMuPDF 文本层（有则用）+ Tesseract（扫描页）→ 分块 → `bge-small` 稠密检索；与主线**共用同一张图、同一回答器、同一 top-k**，只换 retriever。
- **查看器**：Streamlit——证据框叠加 + vision / OCR 并排；金标答案 / 证据页与模型引用另作文本列出。

## 架构

### 数据流

```
已有 PDF ──> PyMuPDF 渲染 (180 DPI)
        │
   ┌────┴──────────────┐
 colSmol 多向量       OCR-first 分支
 (MPS / float16)     PyMuPDF 文本 / Tesseract
   │                      │
 torch MaxSim 索引    bge-small 文本编码 → 分块向量索引
   └────┬──────────────┘
    页面池（去重后的候选页）
        │
   ┌────┴──────────────────────────────────┐
   │  LangGraph agentic 循环                │  assess：池子不够 → 追问，回 retrieve
   │  plan → retrieve → assess ⟲           │  verify：引用不支撑 → 回 retrieve
   │       → synthesize → verify ⟲         │  MAX_ROUNDS 默认 5（检索轮数）
   └────┬──────────────────────────────────┘
        │
  答案 + (doc_id, page) 引用 + bbox
        │
  Streamlit 查看器（叠加 / 并排）
```

### agentic 图

| 节点 | 作用 |
| --- | --- |
| `intent` | 前置分类（共享组件 `intent`）：这一轮是 `chat` 还是 `work`。仅 `ask` / `chat` / REPL 注入 `classifier` 时挂载；`eval` 传 `classifier=None`，**该节点不挂、图与改动前逐字一致**，指标可比。分类器只看最近 3 条有人类文本的 turn（工具消息 / 图片不占额度，单条截 200 字符），整形在组件内完成 |
| `chat` | `chat` 轮：一次普通回答调用，**不检索**、`citations=[]`（由代码保证，不靠模型）→ `END` |
| `plan` | 结构化输出把问题分解成 1..N 个子查询（多跳题 → 多条链） |
| `retrieve` | 每个子查询调 retriever（`MultiVectorRetriever` / `TextRetriever`）→ top-k，去重并入页面池 |
| `assess` | 自省：页面池够不够答？不够 → 产出追问子查询，回 `retrieve` |
| `synthesize` | 回答器读页面池图 + 问题 → 带引用答案 |
| `verify` | 检查每条引用是否支撑结论；不支撑 → 回 `retrieve`；**引用落在检索池外则确定性地判不支撑** |
| `recover` | `MAX_ROUNDS` 用尽 → `recover_exhausted`；零检索 → `recover_empty`；写 `stop_reason` 收口 |

- **有界性**：`rounds` 随 state 持久化，**只在 `retrieve` 自增一次**（= 检索轮数）。`assess` 与 `verify` 只能各把循环推回 `retrieve`，且进 `retrieve` 前一律先判 `rounds < MAX_ROUNDS`，故 `rounds` 恒 `<= MAX_ROUNDS`。若两处各 +1，两处守卫都只按单次自增判断，**实测会冲到 `MAX_ROUNDS + 1`**。
- **零检索不进 `synthesize`**：页面池为空时无页可送，直接走 `recover`——否则答出来的只能是编造。
- **`stop_reason` 只在非正常终止时写**：正常跑完留空。另有三值——`recover_exhausted` / `recover_empty` 由 `recover` 写（**关于语料**），`budget_exhausted` 由节点边界写（撞到调用数 / token / 墙钟任一档，**关于钱包**）。空串即「正常完成」，评测据此归因（难题 vs 索引坏了 vs 预算不够）。
- **`intent` 的 `work_hint` 是承重的**：判据轴是「回复是否需要落地在语料上」，不是「模型是否知道答案」。hint 若只写「需要检索语料库」，模型对**自己能凭参数知识作答**的语料题会判 `chat`（「Where was Gestalt psychology concieved?」实测 8 次 6 次判 `chat` → 静默跳过检索、`citations=[]`、答的是模型记忆而非文档，且与文档口径不同）。显式否定「模型自身知识」并枚举「解释 / 总结 / 翻译 / 比较 / 陈述文档事实」都算 `work` 后，同组题 72/72 判对。**缩句会退化**：语义等价的短句实测 4/8 判错，改这段文字必须重跑判据抽查。
- **state 只放可安全往返的值**（query / 子查询 / page id / 答案文本 / `rounds` / 标量）；页面图像与 torch 张量**不进 state**，以路径或 id 传递。

### 模块边界

均在 `src/multimodal_doc_qa/` 下，`cli.py` 是 Typer 入口：

| 路径 | 职责 | 依赖 |
| --- | --- | --- |
| `render/` | PDF → 页面 PNG | pymupdf |
| `embed/` | 默认 `ColSmol-500M`，可选 `ColQwen2.5-v0.2` | colpali-engine, torch |
| `index/` | torch 张量多向量存储 + MaxSim | torch |
| `retrievers/` | LangChain `BaseRetriever` + 声明式装配：`assembly.py`（臂表 `Index` / `Component` / `Arm`，`MODES` 派生自 `ARMS`）· `multivector.py`（`maxsim`）· `pool.py`（按 patch 平均池化）· `text.py`（`ocr` / `abstract`）· `bm25.py`（手写 BM25 词法臂）· `keywords.py`（`lexical-kw`：LLM 先抽词，仍走同一份 BM25）· `hybrid.py`（RRF 融合）· `rerank.py`（可选重排） | langchain-core, index |
| `agent/` | `plan` / `assess` / `verify` 的 prompt 与结构化 schema | langchain |
| `graph.py` | LangGraph 装配（`intent` / `chat` / `plan` / `retrieve` / …），维护页面池与 `rounds`；`classifier=None` 时前置两节点不挂 | langgraph, langchain |
| `limits.py` | `from_settings`。账本在共享组件 `budget`：调用次数、token、墙钟 | budget |
| `synth/` | 合成 + 引用 / bbox 抽取；`reply()` 为 `chat` 轮的一次普通回答调用（不绑 schema，`citations=[]`） | langchain |
| `abstract.py` | `abstract` 臂：入库时视觉模型写页描述，供文本检索 | — |
| `baseline/` | OCR-first 文本抽取 + 分块 | pymupdf, pytesseract, sentence-transformers |
| `eval/` | 指标、runner、结果落盘 | — |
| `ui/` | `viewer.py`（Streamlit 查看器）· `console.py`（REPL 渲染） | streamlit, repl-console |
| `cli.py` | `ingest` / `ask` / `eval` / `chat` 入口；`_load_deps(..., enable_intent=True)` 注入共享组件 `intent` 的 `Classifier`（模型随 `answerer_model`） | typer, intent |

**图产物**：`graph.png` 由 `python -m multimodal_doc_qa.graph` 生成（`__main__` 给 `GraphDeps` 传非 `None` 的 `classifier` 占位，故图含 `intent` / `chat` 两节点）；拓扑改了重跑；`example.png` 为手工截图。README 只放这两张图。

## 技术栈

- **Python**：共享 venv **3.14.6**；全栈依赖已用 `uv pip install --dry-run` 验证可解析。
- **编排 / 模型抽象**：**LangGraph** 装配 ask 流水线（`langgraph>=1.2.12`）；**LangChain**（`init_chat_model`、`BaseRetriever`，`langchain>=1.4.2` + `langchain-deepseek>=1.1.0`）。
- **模型**：回答器默认 `deepseek:deepseek-flash`（原生多模态，图 ≤384 tok/张），provider 与模型名经 `MDQ_ANSWERER_MODEL` 换；凭据放仓库根 `local.env`。`lexical-kw` 臂的抽词复用同一个 `MDQ_ANSWERER_MODEL`（不新增模型配置）。
- **Embedder**：`colpali-engine==0.3.18` + `torch==2.13.0`（**MPS**）+ `transformers==5.18.0`；默认 `vidore/colSmol-500M`。
- **页面渲染**：`pymupdf==1.28.2`。
- **索引**：`torch==2.13.0`（张量批量 MaxSim；设备 / 精度可配，与 embedder 共用同一 torch）。
- **基线文本检索**：`sentence-transformers` + `BAAI/bge-small-en-v1.5`（`MDQ_OCR_EMBEDDER_MODEL`，只编码 OCR 臂的页文本和查询；设备与视觉编码器同一个 `MDQ_DEVICE`）。
- **OCR**：Tesseract（系统依赖，`brew install tesseract`）+ `pytesseract`。
- **查看器**：`streamlit`。
- **可观测性**：OTel `gen_ai.*` → Langfuse（`LANGFUSE_*`）。
- **依赖唯一来源**：仓库根 `requirements.txt`；项目以 `-e ./projects/multimodal-doc-qa` 加入。

## 交付物

- **CLI**：`doc-qa`（无子命令 → 加载 artifacts 后进 REPL）· `doc-qa chat [--mode]` · `doc-qa ingest <corpus_dir>` · `doc-qa ask "<question>" [--mode]` · `doc-qa eval --questions <questions.json> [--mode] [--out] [--agentic] [--max-tokens] [--max-seconds]`。`ask` / `chat` / REPL 在 `START` 后先过 `intent` 门：`chat` 轮**不检索、不产生引用**（控制台 `sources none`），`work` 轮照常进 `plan → retrieve → …`；`eval` 不挂门；`eval` 默认只跑检索器（`nDCG@k` / `recall@k`，不构造 chat 模型），`--agentic` 才跑完整图并出答案侧指标——唯一例外是 `lexical-kw`：它要在检索时调一次 LLM 抽词，故是纯检索里唯一会建 chat 模型的臂。
- **十条检索臂**（`--mode` 或 `MDQ_MODE`，默认 `maxsim`）：`maxsim` 多向量后期交互 · `pool` 同一份多向量按 patch 平均池化 · `ocr` 页文本 · `abstract` 入库时视觉模型写的页描述 + 文本检索 · `lexical` 手写 BM25（读 `ocr_index` 的块，零嵌入） · `lexical-kw` 同一份 BM25，但查询先经一次 LLM 抽成关键词（模型惰性建于首次查询） · `hybrid-ocr` / `hybrid-abstract` / `hybrid-pool` / `hybrid-maxsim` 各把 BM25（恒读 `ocr_index`）与该 dense 臂按 RRF 融合，两路各取 `4 × top_k` 后归并到页。检索臂由 `retrievers/assembly.py` 的声明表装配，`MODES` 派生自臂表，CLI 与状态栏不再各存一份。会话内切换与状态栏由共享组件 `repl-console` 提供（见根 `README.md` 组件表）。
- **`ingest`**：接受 `pdf` / 图片（`png` / `jpg` / `jpeg` / `webp`）/ `txt` / `md`。PDF 与图片走 `encode_images`；纯文本按块切、用同一视觉编码器的 `encode_texts` 进视觉索引，**不光栅化**。字节相同的后一份文件跳过。视觉索引落 `multivector_index.pt`（`maxsim` / `pool` 共用，由 `vision_index.pt` 改名——升级时把旧文件改名即可，不必重跑 `ingest`）。`abstract` 臂的描述只在 `MDQ_MODE=abstract` 或 `MDQ_ABSTRACTS=1` 时随 `ingest` 写入 `abstract_index.pt`，缓存键是图片字节的 `sha256`。`MDQ_RERANK=1` 时检索后按视觉模型重排，默认关。
- **查看器**：`streamlit run .../ui/viewer.py`——证据框叠加 + vision / OCR 并排，并把金标答案、金标证据页、模型引用（`page_id` + 是否给了 bbox）分别列出。`ask` / REPL 把每一轮写进 `artifacts/turns.jsonl`，**没有 `questions.json` 也能看红框**；蓝框只在那份金标文件存在、且题目 id 对得上时画。
- **评测**：`eval/results.jsonl`——每次 run 由 `kind="run"` 头（commit / 模型 / 日期 / 配置）开头、逐题 `kind="question"` 明细跟写、`kind="summary"` 收尾，**边跑边 flush**，故中途失败（API 报错、超时）时已得结果仍在文件里。逐题行除指标外留原始材料（`ranked` / `pool` / `answer` / `citations` / `stop_reason` / `rounds` / `calls` / `tokens`），使新增指标**无需重跑模型即可回算**。纯检索（默认）下图侧字段（`answer` / `citations` / `pool` / `pool_recall` / `iou_at_threshold` / `bbox_hit_rate` / `rounds` / `calls` / `tokens` / `stop_reason`）写 `null`（键保留），`run` 头带 `retrieval_only` 与 `extractor_model`（后者只对 `lexical-kw` 非 `null`，记它抽词用的模型；其余臂为 `null`）；gold 无 bbox，故 `--agentic` 下 `iou_at_threshold` 与 `bbox_hit_rate` 退化为同一个数（衡量答案是否引到 gold 页，非框位）。注意 `lexical-kw` 的 `ranked` 照记、指标可回算，但**抽出的关键词本身不入行**，审计抽词质量需重跑模型。
- **README.md**：一条命令端到端跑通。

**验证**：单测 `pytest`（MaxSim 对照朴素实现、引用解析、bbox 命中判定、预算熔断、LangGraph 有界性与 state 往返）；冒烟 `transformers 5.18 × colpali-engine 0.3.18` 在 MPS 上跑通一次前向（**落代码前第一件事**）；端到端 `doc-qa ingest` → `doc-qa eval` 跑出非空 `results.jsonl`。
