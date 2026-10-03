# multimodal-doc-qa

- **项目**：multimodal-doc-qa（Agentic RAG 实现）/ 所属 Phase：Phase19 / Capstone 04
- **spec 链接**：[https://aieng-zh.cn/lessons/19-capstone-projects/04-multimodal-document-qa/](https://aieng-zh.cn/lessons/19-capstone-projects/04-multimodal-document-qa/)

## 目标与范围

**目标**：端到端做出一个 **Agentic RAG** 文档问答系统——把 PDF 页面当图像，用多向量
**后期交互（late interaction）**做检索（默认 `vidore/colSmol-500M`）；其上叠一层**有界 agent 循环**（规划 → 多轮检索 → 充分性自省
→ 合成 → 引用自检），用托管 VLM 带引用作答，在查看器里把证据区域叠回原页；并配一条 **OCR-first
文本流水线**跑同一张图做基线。

agency 只改变「取哪些页」，不改变页面表示——检索与证据分项不受影响，多页推理受益。

**MVP（第一版只做这些）**

- **自建合成语料**：程序化生成多页 PDF，五类内容各成页——`paragraph` / `table` / `chart` /
  `handwriting` / `formula`；部分页栅格化为「扫描件」。生成期即产出 ground truth（页号 + 归一化 bbox，0..1）。
- **渲染**：PyMuPDF → 页面 PNG（180 DPI，长边归一化）。PDF 页尺寸由 `PAGE_W/PAGE_H` 反推以
  保持画布长宽比——编码器等比缩放，长宽比变形会被学进去。
- **文本层**：非扫描页在 PDF 中写入**不可见文本层**（`render_mode=3`），等价于数字版页面原生
  自带的可提取文本；扫描页不给。图表页只暴露标签与坐标刻度，**不暴露柱值**——这正是
  vision-first 与 OCR-first 要分胜负的地方。
- **编码预算**：`ColQwen2.5` 处理器 `max_pixels = 602112`（= 768 × 28²）。超过约 0.6 MP 的
  像素在进编码器前即被丢弃，故渲染 DPI 由 OCR 基线 / 查看器需求决定，而非检索质量。
- **编码**：默认 `vidore/colSmol-500M`（MPS / float16）出多向量。换成 `vidore/colqwen2.5-v0.2` 时设 `MDQ_EMBEDDER_MODEL`；加载失败直接报错，没有第二套 checkpoint。
- **索引**：torch 张量精确 **MaxSim**（逐查询 token 取每页 patch 最大点积再求和）。
- **Agentic 循环**：`plan → retrieve → assess ⟲ → synthesize → verify ⟲`；
  `rounds` 只在 `retrieve` 自增（= 检索轮数），超 `MAX_ROUNDS = 3` 或零检索由 `recover` 强制收口。
- **合成**：`deepseek:deepseek-flash` 读**页面池**图 + 问题，`with_structured_output(Answer)` 直接产出带
  `(doc_id, page, bbox?)` 引用的答案——引用与 bbox 在合成时即结构化落地，无独立抽取步骤。
- **证据区域**：引用的 bbox 叠到源页；模型不给 bbox 即为页级引用。
- **装配**：LangGraph 装配；检索经 LangChain `BaseRetriever` 注入。
- **OCR-first 基线**：PyMuPDF 文本层（有则用）+ Tesseract（扫描页）→ 分块 → `bge-small` 稠密检索；
  与主线共用同一张图、同一回答器，只换 retriever。
- **查看器**：Streamlit——证据框叠加 + vision / OCR 并排。
- **评测**：nDCG@5 / recall@k、答案准确率、bbox 命中率、bytes/page、index p95、answer p95、
  每问检索轮数与模型调用数。

**不做**

- Vespa / 托管多向量服务；自托管 `Qwen3-VL-30B` / `InternVL3`；Next.js 15 查看器。
- Nougat / dots.ocr OCR 兜底通道；手写 OCR 流水线横向对比；多租户 / 鉴权 / MCP。
- ViDoRe v3 / M3DocVQA 公开榜对比——用自建语料，不声称与任何公开榜可比。
- 训练 / 微调任何模型（只用现成 checkpoint）。
- single-shot（非 agentic）形态及其消融——只交 agentic 一种形态。代价：answer p95 无同语料基线，
  agency 的边际收益不可量化。

## 架构

### 数据流

```
语料生成器 ──> 多页 PDF + ground_truth.json
                     │
                 PyMuPDF 渲染 (180 DPI)
                     │
              ┌──────┴───────┐
        colSmol 多向量        │   OCR-first 分支
        (MPS/float16)        │   PyMuPDF 文本 / Tesseract
              │               │        │
        torch MaxSim 索引      │   bge-small 文本编码
              │               │        │
              │               │   文本分块向量索引
              └──────┬────────┘
                 页面池（去重后的候选页）
                     │
   ┌─────────────────┴──────────────────┐
   │  LangGraph agentic 循环             │
   │  plan → retrieve → assess ⟲         │  assess：池子不够 → 追问，回 retrieve
   │       → synthesize → verify ⟲       │  verify：引用不支撑 → 回 retrieve
   │                                     │  MAX_ROUNDS = 3（检索轮数）
   └─────────────────┬──────────────────┘
                     │
        答案 +(doc_id,page) 引用 + bbox
                     │
        Streamlit 查看器（叠加 / 并排）
```

### agentic 图

| 节点 | 作用 |
| --- | --- |
| `plan` | 结构化输出把问题分解成 1..N 个子查询（多跳题 → 多条链） |
| `retrieve` | 每个子查询调 retriever（`MultiVectorRetriever` / `TextRetriever`）→ top-k，去重并入页面池 |
| `assess` | 自省：页面池够不够答？不够 → 产出追问子查询，回 `retrieve` |
| `synthesize` | `ChatDeepSeek` 读页面池图 + 问题 → 带引用答案 |
| `verify` | 检查每条引用是否支撑结论；不支撑 → 回 `retrieve`；引用落在检索池外则确定性地判不支撑 |
| `recover` | `MAX_ROUNDS` 用尽 → `recover_exhausted`；零检索 → `recover_empty`；写 `stop_reason` 收口 |

- **有界性**：`rounds` 随 state 持久化，**只在 `retrieve` 自增一次**（= 检索轮数）。
  `assess` 与 `verify` 只能各把循环推回 `retrieve`，进 `retrieve` 前一律先判 `rounds < MAX_ROUNDS`，
  故 `rounds` 恒 `<= MAX_ROUNDS`。若在 `assess` 与 `verify` 各 +1，两处守卫都只按单次自增判断，
  实测会冲到 `MAX_ROUNDS + 1`。
- **零检索不进入 `synthesize`**：页面池为空时无页可送，直接走 `recover`——否则答出来的只能是编造。
- **`stop_reason` 只在非正常终止时写**：正常跑完留空；`recover_exhausted` / `recover_empty` 才写。
  空字符串即「正常完成」，评测据此归因（难题 vs 索引坏了）。
- **state 只放可安全往返的值**（query / 子查询 / page id / 答案文本 / `rounds` / 标量）；
  页面图像与 torch 张量不进 state，以路径或 id 传递。

### 模块边界

| 路径 | 职责 | 依赖 |
| --- | --- | --- |
| `src/multimodal_doc_qa/corpus/` | 合成语料生成（PIL + matplotlib 排版）+ ground truth | pillow, matplotlib, pymupdf |
| `src/multimodal_doc_qa/render/` | PDF → 页面 PNG | pymupdf |
| `src/multimodal_doc_qa/embed/` | 默认 `ColSmol-500M`，可选 `ColQwen2.5-v0.2` | colpali-engine, torch |
| `src/multimodal_doc_qa/index/` | torch 张量多向量存储 + MaxSim | torch |
| `src/multimodal_doc_qa/retrievers/` | LangChain `BaseRetriever`：`MultiVectorRetriever`（vision）/ `TextRetriever`（OCR） | langchain-core, index |
| `src/multimodal_doc_qa/agent/` | `plan` / `assess` / `verify` 的 prompt 与结构化 schema | langchain |
| `src/multimodal_doc_qa/graph.py` | LangGraph 装配，维护页面池与 `rounds` | langgraph, langchain |
| `src/multimodal_doc_qa/budget.py` | 单次 ask 的调用数 / token / 墙钟三档熔断与用量统计 | langchain-core |
| `src/multimodal_doc_qa/synth/` | `deepseek:deepseek-flash` 合成 + 引用/bbox 抽取 | langchain |
| `src/multimodal_doc_qa/baseline/` | OCR-first 文本抽取 + 分块（供 `TextRetriever` 建索引） | pymupdf, pytesseract, sentence-transformers |
| `src/multimodal_doc_qa/eval/` | 指标、runner、结果落盘 | — |
| `src/multimodal_doc_qa/ui/` | Streamlit 查看器 | streamlit |
| `src/multimodal_doc_qa/cli.py` | `ingest` / `ask` / `eval` 入口 | typer |

**图产物**：`graph.png` 由 `python -m multimodal_doc_qa.graph` 生成（这条命令会先写图，再加载编码器、把语料页建进内存索引、真的问一题），拓扑改了重跑；`example.png` 为手工截图。README 只放这两张图。

## 技术栈

- **Python**：共享 venv **3.14.6**；全栈依赖已用 `uv pip install --dry-run` 验证可解析。
- **编排 / 模型抽象**：**LangGraph** 装配 ask 流水线（`langgraph>=1.2.12`）；**LangChain**
  （`init_chat_model` / `ChatDeepSeek`、`BaseRetriever`，`langchain>=1.4.2` + `langchain-deepseek>=1.1.0`）；
  回答器 `deepseek:deepseek-flash`（DeepSeek V4.1-Flash，原生多模态，图 ≤384 tok/张）。
- **Embedder**：`colpali-engine==0.3.18` + `torch==2.13.0`（**MPS**）+ `transformers==5.18.0`；
  默认 `vidore/colSmol-500M`。`vidore/colqwen2.5-v0.2` 用 `MDQ_EMBEDDER_MODEL` 指定，加载失败不降级。
- **页面渲染**：`pymupdf==1.28.2`。
- **索引**：`torch==2.13.0`（张量批量 MaxSim；设备 / 精度可配，与 embedder 共用同一 torch）。
- **基线文本检索**：`sentence-transformers` + `BAAI/bge-small-en-v1.5`（`OcrEmbedder` / `MDQ_OCR_EMBEDDER_MODEL`，只编码 OCR 臂的页文本和查询；设备与视觉编码器同一个 `MDQ_DEVICE`）。
- **OCR**：Tesseract（系统依赖，`brew install tesseract`）+ `pytesseract`。
- **查看器**：`streamlit`。
- **可观测性**：OTel `gen_ai.*` → Langfuse（`LANGFUSE_*` 已在 `local.env`）。
- **依赖唯一来源**：仓库根 `requirements.txt`；项目以 `-e ./projects/multimodal-doc-qa` 加入。

## 指标与基线

- **主指标**（逐内容类型 × 逐范式报告，不报单一平均值）：
  - **检索质量**：nDCG@5（vision-first vs OCR-first），另记 recall@k（按逐轮累计的页面池算）。
    **实现状态**：`ndcg_at_k` 已实现并接入 `eval`；**`recall@k` 未实现**。
    nDCG 的输入**必须是检索器自己的排序**（`retriever.invoke(question)`），**不能**用图 state 的
    `page_ids`——那是按子查询顺序 append 的、非分数序，对它算 rank 类指标没有意义。
    且输入**必须先按页去重**：OCR 臂会返回同一页的多个 chunk，同一页被计多次而 IDCG 只算一次，
    实测曾得出 **nDCG@5 = 1.1632**（>1，不可能）。`ndcg_at_k` 内部已去重（在 k 截断之前）。
  - **答案准确率**：自建 **100 问多页 holdout**（每题需 ≥2 页证据）的准确率。
    **实现状态**：未实现。**判对错的口径已定：确定性包含匹配**（归一化后 gold 答案的数值/词元
    出现在回答文本里），指标名为 `answer_containment` 而**不叫 accuracy**——它宽容（模型啰嗦也能过），
    名字必须体现这一点。不用 LLM-as-judge（成本 + 方差，且需要先校准）。
  - **证据区域落地**：被引用区域中实际包含答案区间的占比（bbox 命中率）。
    **实现状态**：已实现两项，**主指标是 `iou_at_threshold`（默认 IoU ≥ 0.5）**，
    同时保留 `bbox_hit_rate`（严格包含）作参照。阈值记入每次 run 的 provenance
    （`--iou-threshold` 可改）——脱离阈值的 IoU 数字不可比较。
    改用 IoU 作主指标的原因见已知坑 21：严格判据会**反向**排序证据框。
  - **存储 / 延迟**：bytes/page、index p95、answer p95（agentic）、
    每问检索轮数与模型调用数分布。
    **实现状态**：answer 延迟 p50/p95、rounds / calls / tokens 分布**已随逐题行落盘**；
    bytes/page 与 index p95 **未实现**。
  - **评测集状态**：声明的 ~300 页 + 100 问多页 holdout **尚未生成**。当前语料为
    **6 页 / 8 问**（对应 `generate_corpus(n_docs=2, seed=42)`），其中 6 问为单页证据、
    2 问才满足「≥2 页证据」。故现有 `results.jsonl` 里的数字**只能证明流水线通、不能引用为结论**。
    另：`_build_questions` 的「每页 1 个单跳题」本身就不满足「每题需 ≥2 页证据」，放大语料时需一并修。
- **基线**：OCR-first 文本流水线——**同一张 agentic 图**，只把 retriever 换成 `TextRetriever`；
  语料、问题、循环、回答器、top-k 全相同。
- **数据集**：自建 ~300 页（五类 × 各若干页）+ 100 问多页 holdout；holdout 生成后冻结，
  调参只看 dev 集。
- **可证伪假设**（写成测试门槛，不达标即失败）：
  - H1：表格 / 图表 / 手写三类上，vision-first nDCG@5 比 OCR-first **高 ≥ 0.10**；
    纯段落类上两者差异 **< 0.05**。
  - H2：~300 页下 index p95 **< 100 ms**。
  - H3：holdout 上 `rounds` 分布可解释（单跳题多为 1 轮、多跳题 >1 轮），且被 `recover`
    强制收口的比例 **< 10%**。
- **落盘**：`eval/results.jsonl` —— 每次 run 由**一行 `kind="run"` 头**（commit / 模型 / 日期 / 配置）开头、
  逐题 `kind="question"` 明细跟写、`kind="summary"` 聚合行收尾。行是**边跑边 flush** 的，
  故中途失败（API 报错、超时）时已得结果仍在文件里。
  逐题行除指标外还留原始材料（`ranked` / `pool` / `answer` / `citations` / `stop_reason` /
  `rounds` / `calls` / `tokens`），使新增指标无需重跑模型即可回算。
- **只与自己比**：不声称与任何公开榜单或外部系统横向胜负。

## 预算

| 维度 | 上限 | 熔断行为 |
| --- | --- | --- |
| 单次 `ask` 模型调用 | **16 次**（`plan` 1 + 每轮 `assess`/`synth`/`verify` 各 1，`max_rounds=5`） | 就地收尾、保留已付费答案、写 `stop_reason=budget_exhausted` |
| 单次 `ask` 检索轮数 | `MAX_ROUNDS = 5` | `recover` 强制收口并写 `stop_reason` |
| 单次 `ask` token | 200k tokens（含图；一次两轮视觉问答实测约 30k，三轮曾顶穿旧的 80k） | 同调用数：就地收尾 + `budget_exhausted` |
| 单次 `ask` 墙钟 | 120 s | 同调用数：就地收尾 + `budget_exhausted` |
| 单次全量评测 | 120 min / 累计 token 上限（`--max-seconds` / `--max-tokens`） | 中止并落盘已得结果 |

- 调用数上限跟轮数走：`1 + 3 × max_rounds`。`max_rounds=3` 时最坏路径（`verify` 每轮打回）实测 **10**，
  `assess` 每轮要页实测 **6**，单跳下限 **4**。三者均有单测钉住（`tests/test_graph.py`）。旧的「8 次」低估了一轮，
  原因是当时用假 synthesizer 测量、每轮漏计一次真实调用；旧公式里的 `bbox 1` 是**幽灵节点**
  （`synthesize` 改为结构化输出后已无独立的 bbox 抽取调用）。
- 记账来源：`usage_metadata.total_tokens`（由 `BudgetCallback` 累加）。
  **不做人民币成本核算**——本项目没有接入单价表，故此前的「30 元 / `--max-cost`」声称已废除，
  熔断只靠 token 与墙钟两档，不假装有货币口径。
- 调用数在**节点边界**计（不依赖 callback）；token 依赖 `BudgetCallback` 被绑上，
  绕过 `build_graph` 自建节点会让 token 档失效（见已知坑 18）。
- 本地 embedder 不计费，但记录 wall-clock。

## 交付物

- **CLI**：`doc-qa`（加载 artifacts 后进入 REPL）、`doc-qa chat [--mode]`、`doc-qa ingest <corpus_dir>`、`doc-qa ask "<question>"`、`doc-qa eval [--mode] [--max-tokens] [--max-seconds]`。
- **CLI 细节**：
  - REPL（不带子命令）：两条索引启动时都加载，`--mode` 只决定开场停在哪条；提示符 `you ›`，下面状态栏是当前路径与编码器；`Shift-Tab` 切 vision / ocr，不重新加载模型；空行忽略；`:q` / Ctrl-C / Ctrl-D 退出，一轮还在跑时 Ctrl-C 只取消这一轮。
  - `ingest --corpus` 接受 `pdf` / 图片（`png` / `jpg` / `jpeg` / `webp`）/ `txt` / `md`：PDF 与图片走 `encode_images`；纯文本按块切、用同一视觉编码器的 `encode_texts` 进视觉索引，不光栅化。字节相同的后一份文件跳过。只有一份文件时 `doc_id` 是词干，同词干有两份则用完整文件名、两份都入库。
  - 检索在 top-k 之后还过分数线：低于本轮最高分 `MDQ_MIN_SCORE_RATIO`（默认 `0.5`）的页不进结果。引用面板只打印这份材料真正有的文件类型（文本页不出现 pdf / png）。
  - CLI 自己读仓库根 `local.env`（环境里已有的同名变量不会被盖掉）；`ask` / REPL / `eval` 调 DeepSeek，需要 `DEEPSEEK_API_KEY`。
- **查看器**：`streamlit run ...`——证据框叠加 + vision / OCR 并排。
- **评测**：`eval/results.jsonl` + 一份对照报告（内容类型 × 范式矩阵）。
- **`outputs/skill-doc-qa.md`**：描述交付物与如何复现。
- **README.md**：一条命令端到端跑通 + 已知限制。

## 验收标准（rubric）

| 权重 | 标准 | 本项目度量方式 |
| --- | --- | --- |
| 25 | 检索 / 问答准确率 | 自建语料上 nDCG@5 与多页问准确率，**对照 OCR-first 基线**（不对比公开榜） |
| 20 | 证据区域落地 | bbox 命中率（引用区域含答案区间的占比） |
| 20 | 存储与延迟工程 | bytes/page、index p95、answer p95（agentic） |
| 20 | 多页推理 | 100 问多页 holdout 准确率 |
| 15 | 来源核查体验 | Streamlit 叠加保真度、并排对比可用性 |

> 与 spec 的差异：spec 的 25 分项写「ViDoRe v3 / M3DocVQA + 公开榜对比」，本项目改用自建语料 +
> OCR 基线，**不声称公开可比**。

**硬拒绝项**

- 用硬编码 / 绕过真实流水线的方式产出「结果」（假数据、预置答案）。
- 评测时挑题 / 换题集，或把基础设施失败计为通过。
- 上报 nDCG 时用不同语料 / 不同问题集对比两种范式。

## 测试与验证

- **单测**：`pytest` —— 索引 MaxSim 正确性（对照朴素实现）、
  语料生成器 ground-truth 一致性、引用解析、bbox 命中判定、预算熔断；LangGraph 图
  （retriever 注入切换 vision/OCR、`MAX_ROUNDS` 有界性——用反复要页的假 retriever 断言不超界、
  `stop_reason` 必写、state 往返不丢标量）。
- **冒烟**：`transformers 5.18` × `colpali-engine 0.3.18` 在 MPS 上跑通一次前向（**第一件事**）。
- **端到端**：`doc-qa ingest` → `doc-qa eval` 跑出非空 `results.jsonl`。
- **双向可证伪**：每类内容都有一条「vision 赢」和一条「段落类打平」的断言，任一不成立即暴露构造问题。

## 风险

### 未决

1. **transformers 5.x 兼容性**：`colpali-engine 0.3.18` 的实测上限未知；若 API 断裂，需钉
   `transformers` 到兼容版本或改用 `MultiVectorEncoder` 高层 API。**落代码前先冒烟。**
2. **MPS 上的 4B**：`vidore/colqwen2.5-v0.2` + 768-patch 前向在 16GB（可 wired ≈10.7GB）下可能 OOM。
   默认已是 `vidore/colSmol-500M`。若显式换上 4B：batch=1、`torch.mps.empty_cache()`，必要时降 patch 上限。
3. **Tesseract 缺失**：系统未装则 OCR 基线跑不动；备选是「渲染期文本层」当完美 OCR（更保守）。

### 已知坑

4. **MPS 无 `flash_attention_2`**：`attn_implementation` 留空 / eager；`bfloat16` 若算子不支持则退
   `float16`。别照抄 CUDA 教程。
5. **`colpali-engine` prompt 漂移**：0.3.9 / 0.3.11 / 0.3.13 改过 query / document prompt；
   checkpoint 与引擎版本必须配套，否则 embedding 分布偏移。
6. **MaxSim 设备选择**：纯内积，MPS 未必快过 CPU；设备写进配置，实测后再定（也影响 fp16 累加精度）。
7. **合成语料不能太容易**：页面若都是干净印刷体，OCR 会打平甚至取胜，H1 证伪。表格要有合并单元格、
   图表无文本层、手写用真实手写字体、公式用 mathtext。
8. **`deepseek-flash` 引用格式**：模型不保证稳定吐 bbox；需结构化输出 + 校验，抽不到就退化为页级，
   并如实计入 bbox 命中率。**且必须关思考模式**（`extra_body={"thinking": {"type": "disabled"}}`）：
   思考模式拒绝一切强制 `tool_choice`，而 `with_structured_output` 默认设 `tool_choice="any"`，
   不关则每次 ask 都 400；`strict=True` / 指定函数名 / `reasoning_effort` 均无效，只有关思考
   或改走 `json_mode`（后者要求 prompt 里出现 `json` 一词）。
8b. **`doc_id` 会编造**：不给页面标签时模型自造 `doc_id`（实测出现 `'image'` / `'test'` / `'unknown'`），
   而 `Citation.doc_id` 只是 `str`，编造值能通过 schema 校验、静默污染引用准确率。
   `page_ids` 必须在图前带标签；引用与检索结果的一致性校验留给 `verify` 节点。
9. **结果可复现**：合成语料带随机种子；评测记录 seed / commit / 模型版本。
10. **`BaseRetriever.invoke` 是同步接口**：torch / MPS 逻辑须在同一线程内，节点里不要跨线程复用
    MPS 上下文。
11. **graph state 只放可安全往返的值**：PIL / torch 对象经 state 往返会丢类型；页图与索引张量只以
    路径 / id 传递，需要持久化的标量进显式 state 字段。
12. **agentic 循环不收敛**：`assess` / `verify` 若反复要页会烧钱。靠 `MAX_ROUNDS=3` + `rounds` 进
    state 兜底；边界用 mock retriever 单测，跑批用预算熔断兜底。
13. **引用自检可能反倒拖低准确率**：`verify` 是 LLM，误判会把好答案打回重取。需在 dev 集上调 prompt；
    若净收益为负，如实报告。
14. **MPS 权重加载必须同步**：transformers 5.x 默认用 `ThreadPoolExecutor` 并行 materialize 权重，
    多线程往 MPS 拷贝会**段错误**（Metal 分配非线程安全），崩在 `core_model_loading._materialize_copy`。
    修法：加载前设 `HF_DEACTIVATE_ASYNC_LOAD=1`（测试由 `tests/conftest.py` 兜，生产在 encoder 的
    `_load` 里按 `device == "mps"` 兜）。**不要**误当成版本兼容问题去 pin `transformers`。
    退路：`device_map="cpu"` 加载后主线程 `.to("mps")`。
15. **`vidore/colqwen2.5-v0.2` 在 MPS 上加载极慢**：4B，本机 MPS + fp16 下加载 + 编码 6 页
    **实测 >10 分钟未完成**（主动中断）；`vidore/colSmol-500M` 同条件 7.0s + 9.5s。
    因此默认就是 colSmol。要用 4B 必须显式设 `MDQ_EMBEDDER_MODEL=vidore/colqwen2.5-v0.2`。
16. **小语料上页面池会饱和，使 recall 指标虚高**：各子查询的 top-k 结果并池累积。
    实测 6 页语料 + `top_k=5`：单条子查询即覆盖 83%，2 轮后池子 = 整个语料（6/6）。
    此时「recall@k 按累计页面池算」平凡接近 1。评测语料须足够大（~300 页下 `top_k=5` 仅占 1.7%），
    且**不要把 demo 的池子大小当检索质量读**。
17. **索引与编码器之间没有任何绑定（已接受的已知风险）**：`MultiVectorIndex.save` 只存
    `{page_id: tensor}`，不记录写它的 checkpoint / dim / patch 数；`load` 也不校验。
    维数不同会在 MaxSim 处报 matmul 错（还能发现），维数恰好相同则**静默给出错误分数**（发现不了）。
    当前对策只有**纪律**：`ingest` 会打印实际加载的 checkpoint 类名，换模型后**必须重新 ingest**。
    若日后要根治，最小修法是在 `save` 里带上 encoder 标识与 `[dim, patches]`，`load` 不匹配即报错。
18. **`ask` 的 token 预算依赖 callback 被绑上**：调用数在节点边界计（不依赖 callback），但 token
    只能由 `BudgetCallback.on_llm_end` 从 provider 的 usage 里累加。`build_graph` 已把 callback 绑在
    编译图上（`with_config`），实测真机生效（`tokens 13099/80000`）。但若有人绕过 `build_graph`
    自己调节点，token 档会静默失效（恒为 0），只剩 calls 与 seconds 两档兜底。
19. **一次 ask 一个编译图**：预算按图的生命周期计。复用同一张编译图跑第二次 ask，第二次会接着
    上一次的 tally 继续扣，可能一开场就 `budget_exhausted`。跑批时必须每题新建图（或新建 Budget）。
20. **`recover_exhausted` 与 `recover_empty` 语义不对称**：`recover_empty` 走得通「池子为空 → 不喂答案器」，
    **必无答案**；而 `recover_exhausted` 只有一条来路——`verify` 判出未支持项且 `rounds` 已用尽——
    此时 `answer` **已经生成**，故它**总带答案**。实测 8 问里 3 问为此状态且全部有答案。
    于是 `stop_reason=recover_exhausted` 读起来像「放弃了」，实际含义是「答了但校验未通过、且轮数已尽」。
    H3 若要统计「被 recover 强制收口的比例」，**不能**把这两个原因混在一个分母里，否则会把
    「已交付答案」算成「失败收口」。若要根治，应拆出一个独立原因（如 `verify_exhausted`）。
21. **`bbox_hit_rate` 的严格包含判据会反向排序（实测，已改主指标）**：`BBox.contains` 是全有或全无，
    于是**糊一个巨大的框**得 1.0、**框得几乎精准但差 0.005** 得 0.0。实测 8 问：
    `doc000-p0` 模型框 0.670×0.100 页尺寸（gold 仅 0.237×0.023 的一行）记 **1.0**；
    `doc001-p2` 模型框与 gold 几乎重合（IoU **0.689**）因 `x1` 差 0.0048 记 **0.0**。
    均值为 0.375，但**排序是反的**。
    对策：新增 `BBox.iou` 与 `eval.metrics.iou_at_threshold`（默认阈值 0.5）作**主指标**，
    严格包含降为参照并同时落盘。两者在真实数据上是两个不同的数：
    vision IoU@0.5 **0.25** / strict **0.4375**；ocr IoU@0.5 **0.375** / strict **0.50**。
    **不要**把 strict 的低值直接读成「引用区域差」。
22. **8 题规模下，run 间方差足以翻转手臂排序（实测）**：同 8 题、同配置跑两次，
    严格包含指标 vision `0.375 → 0.4375`、ocr `0.25 → 0.50`——**排序反转**；
    同时 `recover_exhausted` 计数从 vision `3` / ocr `1` 变成 vision `1` / ocr `3`，**正好互换**。
    原因：循环里有 3 处 LLM 决策（`plan` / `assess` / `verify`），温度 > 0，且 8 题的分母太小。
    结论：**现有语料上的任何指标都不能用来排序两条手臂**，与用哪个指标无关。
    要下 H1 结论必须先放大语料（~300 页 / 100 问），并比较**同一次 run** 的两臂
    （或每题重复采样后报区间），而不是拿两次 run 的数字对拼。
23. **`eval` 每问多做一次检索**：逐题行里的 `ranked` 需要**检索器自己的排序**，
    而图 state 的 `page_ids` 是子查询顺序、非分数序，取不到分数。故 `cli.run_question`
    额外调一次 `retriever.invoke(question.text)`，代价是每问多一遍查询编码。
    这**不是**可以随手省掉的浪费：省掉就没有可用的排序，nDCG 会退化成对无序列表算 rank。
    若评测成本成为问题，正解是把 `ScoredPage.score` 写进 state（retriever 已经返回它）。
