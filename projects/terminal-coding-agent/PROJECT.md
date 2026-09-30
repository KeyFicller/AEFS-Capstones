# Terminal-coding-agent

- **项目**：terminal-coding-agent / 所属 Phase：Phase19 / Capstone 01
- **spec 链接**：[https://aiengineeringfromscratch.com/lesson?path=phases%2F19-capstone-projects%2F01-terminal-native-coding-agent](https://aiengineeringfromscratch.com/lesson?path=phases%2F19-capstone-projects%2F01-terminal-native-coding-agent)

## 目标与范围

**目标**：端到端做出一个 Coding Agent——输入是 CLI，输出是 pull request——并在 SWE-bench Pro 上跟 mini-swe-agent 比一比。

**MVP（第一版只做这些）**：一个标准 LangGraph 项目，能被 `harbor run -a langgraph` 在沙箱容器内调起，对 **1 个**真实 SWE-bench Pro 任务产出 patch 并通过其自带验证器。

- `langgraph.json` + `make_graph(config)` 工厂 + 含 `messages` 的 state schema
- plan / act / observe / recover 四段循环
- 6 个工具：`read_file`、`edit_file`（带 diff 预览）、`ripgrep`、`tree_sitter_symbols`、`run_shell`（带 timeout）、`git`；每次输出截断至 4k tokens
- ≥4 hook：`PreToolUse`（破坏性命令守卫）、`PostToolUse`（token 记账）、`SessionStart`（预算初始化）、`Stop`（写 trace）
- 三层预算熔断 + `PreCompact`
- `gen_ai.*` OTel span
- 端到端跑通 **1 题**（不是 30 题）

**不做（延后到加固 / 交付阶段）**

- TUI（三栏：plan / 工具流 / 实时预算）—— 评测路径不经过它，只关系 rubric 的开发者体验 15 分
- PR 发布（`git push` + GitHub API 开 PR）
- 30 题完整跑批 + mini-swe-agent 基线
- Terminal-Bench 2.0、自建 holdout、多模型扫描
- MCP StreamableHTTP transport（见风险 2）

## 架构

`plan -> act -> observe -> recover` 四段：

- **Plan**：维护 TodoWrite 风格的状态对象，模型每轮重写它
- **Act**：分派工具调用（读、改、运行、搜索、git）
- **Observe**：捕获 stdout / stderr / 退出码，截断后把摘要喂回
- **Recover**：处理工具错误，既不撑爆上下文，也不无限循环

**Hooks**：`PreToolUse`、`PostToolUse`、`SessionStart`、`SessionEnd`、`UserPromptSubmit`、`Notification`、`Stop`、`PreCompact`——可配置扩展点，运营方在此注入策略、遥测、护栏。

**沙箱由 Harbor 提供**：graph 与工具都在容器内执行，宿主文件系统不可达 → spec 的 hard reject「不许在宿主机执行 git」自动满足。

### Harbor 接入契约（0.23.0 源码核实，即要写的接口）

| 环节 | 契约 |
| --- | --- |
| 注册 | 项目根 `langgraph.json` → `{"graphs": {"<name>": "./file.py:<attr>"}}`；`<attr>` 可为已编译 graph，或工厂 `make_graph(config)`；Harbor `project_path` 默认为 cwd，须显式指向本项目目录 |
| 调用 | `graph.ainvoke({"messages": [...]}, config={"configurable": {...}})`；state **必须含 `messages`** |
| 返回 | 最终答案取 `result["messages"][-1].content` |
| 模型注入 | `model` / `model_kwargs` / `thread_id` 都在 `configurable` → 从 config 读，**禁止在 import 期固化** |
| worktree | Harbor **不**默认注入 `configurable.worktree`；切片约定缺失时回退 `Path.cwd()`（任务工作目录，不是 `/installed-agent/langgraph-project`） |
| token 记账 | runner 自动累加 `result["messages"]` 的 `usage_metadata` |
| 运行位置 | 项目 copytree 进容器 `/installed-agent/langgraph-project`；容器内 venv 为 `/opt/harbor-langgraph-venv`，**Python 3.12**；**不会**带上仓库根 `requirements.txt`，项目须可自安装 |

## 技术栈

- **Python**：本机共享 venv 3.14.6，**容器内 3.12** → 代码须 3.12 兼容
- **编排**：LangGraph + LangChain（model / tool / retriever 抽象）
- **模型**：`langchain-deepseek>=1.1.0`，默认 `deepseek-v4-flash`，provider 与模型名经 `configurable` 注入
- **搜索**：ripgrep 子进程 + tree-sitter（预编译）
- **沙箱 / 评测**：**Harbor 0.23.0**（`uv tool install harbor`，落在 `~/.local/bin`，**默认不在 PATH**）；`--env docker` 为本地默认，`daytona` / `e2b` / `modal` / `runloop` 为云端备选
- **基线**：Harbor 内置 `-a mini-swe-agent`
- **可观测性**：OTel `gen_ai.*` 语义约定 → 自托管 Langfuse
- **PR 发布**：细粒度 token 的 GitHub App，作用域限目标仓库（MVP 后补）

## 指标与基线

- **主指标**：`pass@1`（30 题中首次运行即通过的比例）。**禁止在重试之间 `git reset --hard` 刷分**。同批记录 `turns/task`、`tokens/task`（in/out 分开）、`$/task`。
- **基线**：`mini-swe-agent`，**必须同模型、同 30 题**；模型不同则测的是模型差距而非 harness 差距，对比作废。Live-SWE-agent 仅作上下文参照，不参与打分。
- **数据集**：SWE-bench Pro V2 的 python 子集 **30 题**；抽样规则固定（instance_id 列表 + seed）落 `eval/tasks.json`，否则 matched subset 不成立。每题镜像 `ghcr.io/scaleapi/swe-bench_pro-v2:<instance_id>`（公开可拉，**linux/amd64**）。
- **落盘**：Harbor 原生输出 `pass_at_k` / `cost_usd` / token 统计到 `<job>/result.json`，另有 agent 轨迹、verifier 输出、`trial.log`（`harbor view <job>` 看轨迹）；汇总进 `eval/results.jsonl`，基线同 schema 单独落盘以便逐题配对。

## 预算

| 维度 | 上限 | 熔断行为 |
| --- | --- | --- |
| 轮数 | 50 turns | `Stop` hook → 写 trace |
| 上下文 | 200k tokens | `Stop` hook → 写 trace |
| 成本 | $5.00 / task | `Stop` hook → 写 trace |

- **turns 口径 = 模型调用次数（LLM invocations）**，不是图循环轮数。planner 调用计 1，executor 每次调用各计 1。选此口径的唯一理由：`mini-swe-agent` 的 step 计数同义，基线对比的 `turns/task` 才可比。

- `PreCompact` @150k tokens：把旧轮次摘要成 prior-state block，腾出空间但不丢计划。
- 记账来源：`usage_metadata` × 模型单价；**`cache_read_tokens` 必须计入**——DeepSeek 有前缀缓存，漏算会让 `$/task` 偏高，且重写历史会令缓存失效。
- **禁止无预算上限运行**：开放式运行会污染评测对比。
- 30 题 × $5 = **$150 上限**（仅模型成本）。沙箱容器分钟数与镜像拉取（每题数 GB）**另计**，用完即删；本地 docker 近零成本，但 16GB 内存把并发压到约 1。

## 交付物

**Harbor 垂直切片（已验证 2026-09-30）**

- 手写任务 `harbor_tasks/greeter-fix` + 项目根 `langgraph.json` + `requirements-harbor.txt`；`harbor run -a langgraph` 在 Docker trial 内调起本项目 `make_graph`，verifier `reward=1`（**不是** SWE-bench Pro 进度）。
- 仓库根复现命令（cwd = 仓库根；job 落在仓库根 `jobs/`，该目录 gitignore）：
  ```bash
  export PATH="$HOME/.local/bin:$PATH"
  set -a; source local.env; set +a
  harbor run \
    -p projects/terminal-coding-agent/harbor_tasks/greeter-fix \
    -a langgraph \
    -m deepseek:deepseek-v4-flash \
    --ae DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY" \
    --allow-agent-host api.deepseek.com \
    -k 1 \
    --ak project_path="$(pwd)/projects/terminal-coding-agent"
  ```
  - **凭据必须用长选项 `--ae`**：短写 `-ae` 会被 Click 拆成 `-a/-e`，把 `KEY=value` 当多余参数并可能把 key 打进错误信息。
  - `langgraph` 未声明 `MODEL_CONNECTION`，Harbor **不会**自动转发 `DEEPSEEK_API_KEY`；`--env-file` 只喂宿主机 harness，不进容器。
  - 必须显式 `--ak project_path=...`（默认是 cwd，不是本项目目录）。
  - `--allow-agent-host api.deepseek.com` 固定带上；任务 `network_mode=public` 时只 warning 并被忽略，`allowlist` 时则必需。
  - worktree：Harbor 不默认注入 `configurable.worktree`；缺失时 `make_graph` 回退 `Path.cwd()`（任务工作目录）。

**MVP（尚未完成）**

- `langgraph.json` + graph（四段 + ≥4 hook + 6 工具）+ 含 `messages` 的 state schema
- 目标命令形态（真实 SWE-bench Pro 任务；flags 与切片相同，路径/`-i` 换成实例）：
  ```bash
  export PATH="$HOME/.local/bin:$PATH"
  set -a; source local.env; set +a
  harbor run -p <v2/tasks> -a langgraph -m deepseek:deepseek-v4-flash \
    --ae DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY" \
    --allow-agent-host api.deepseek.com -k 1 -i <instance_id> \
    --ak project_path="$(pwd)/projects/terminal-coding-agent"
  ```
- 验证证据：1 个真实 SWE-bench Pro 任务的完整 job 目录（`result.json` + agent 轨迹 + verifier 输出），且 `pass@1 = 1` 或给出失败模式分析

**完整交付（MVP 后补齐）**

- CLI `agent run <repo> "<task>"` + TUI 三栏
- `eval/tasks.json`、`eval/run_eval.py`、`eval/results.jsonl` + 匹配的 mini-swe-agent 基线运行
- ≥5 次完整运行的 OTel trace 归档；100% 工具调用带 span
- 报告：哪些题 harness 能解而基线不能（反之亦然）；结尾写前三大失败模式与对应 hook 改动
- PR 发布（正文含 plan 与 diff 摘要；**禁止直推 main**）
- `outputs/skill-terminal-coding-agent.md`

## 验收标准（rubric）

| 权重 | 标准 | 度量方式 |
| --- | --- | --- |
| 25 | SWE-bench Pro pass@1 | 匹配的 30 题子集 vs mini-swe-agent 基线 |
| 20 | 架构清晰度 | plan/act/observe 分离、hook 面、工具 schema 可读性 |
| 20 | 安全 | 沙箱逃逸红队 + 破坏性命令守卫审计 |
| 20 | 可观测性 | 100% 工具调用带 span、逐轮 token 记账 |
| 15 | 开发者体验 | 冷启动 < 2s、崩溃恢复、Ctrl-C 取消语义 |

> MVP 覆盖架构 20 + 安全 20 + 可观测性 20，以及 pass@1 的必要条件（能产出 patch）；开发者体验 15 属「不做」。

**硬拒绝项**

- harness 在宿主机文件系统上直接调 git，而非沙箱内执行
- agent 能写出 worktree 之外，或在无显式 allowlist 时 curl 外部 URL
- 上报评测数字没有在同一批 30 题上跑匹配基线
- 「通过率」依赖重试之间 `git reset --hard`

## 风险

**未决**

1. **执行器本地还是云端**：MVP 用本地 docker。已实测 Rosetta 下 amd64 仅 **1.39×** 开销（容器内 `vendor_id: VirtualApple`），性能可行；16GB 内存限制并发 ≈ 1，30 题跑批时再定是否上云（只剩「花不花钱」一个变量）。
2. **MCP StreamableHTTP vs 进程内工具**：spec 要求经 MCP 暴露工具，规则要求用 LangChain 做 tool 抽象，两者不同层（wire protocol vs 进程内）。本地同机评测下 MCP 是纯开销，但 rubric 的「工具 schema 可读性」偏向显式 schema。**未决。**
3. **工具面口径**：技术栈栏写 ripgrep + tree-sitter，spec 要求 6 个工具。MVP 按 6 个实现，口径需与实现对齐。

**已知坑**

4. **`strict` 结构化输出的 schema 约束**：DeepSeek `with_structured_output(strict=True)` 要求 schema 中所有 object 属性标 `required`——计划 schema 不能含可选字段；该参数还可能被 API 忽略，当 best-effort 处理，保留校验边界。
5. **`PreCompact` 与消息重建**：摘要 / 裁剪会重写历史；若从纯文本重建消息会丢 `AIMessage.additional_kwargs`，在 thinking 模式触发 400 → 保持原始消息对象，勿重建为 `AIMessage(content=...)`。
6. **模型 ID 时效**：只用 `deepseek-v4-flash` / `deepseek-v4-pro`；`deepseek-chat` / `deepseek-reasoner` 已于 2026-07-24 停用。provider 文档常滞后，勿照抄。
7. **解释器版本错配**：本机 3.14.6 上 `--dry-run` 依赖全部解析通过，**不能**外推为容器 3.12 内可安装；落代码前先在 3.12 下验一次依赖解析。
