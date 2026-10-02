# Terminal-coding-agent

- **项目**：terminal-coding-agent / 所属 Phase：Phase19 / Capstone 01
- **spec 链接**：[https://aiengineeringfromscratch.com/lesson?path=phases%2F19-capstone-projects%2F01-terminal-native-coding-agent](https://aiengineeringfromscratch.com/lesson?path=phases%2F19-capstone-projects%2F01-terminal-native-coding-agent)

## 目标与范围

**目标**：端到端做出一个 Coding Agent——输入是 CLI，输出是一份 patch——在隔离 worktree 内读改代码、跑命令、自我验证。

**MVP（第一版只做这些）**：一个标准 LangGraph 项目，能被 `harbor run -a langgraph` 在沙箱容器内调起，对 **1 个**手写 Harbor 任务产出 patch 并通过其自带验证器。

- `langgraph.json` + `make_graph(config)` 工厂 + 含 `messages` 的 state schema
- plan / act / observe / recover 四段循环
- 6 个工具：`read_file`、`edit_file`（带 diff 预览）、`ripgrep`、`tree_sitter_symbols`、`run_shell`（带 timeout）、`git`；每次输出截断至 4k tokens
- ≥4 hook：`PreToolUse`（破坏性命令守卫）、`PostToolUse`（token 记账）、`SessionStart`（预算初始化）、`Stop`（写 trace）
- 三层预算熔断 + `PreCompact`
- `gen_ai.*` OTel span
- 端到端跑通 **1 题**（`harbor_tasks/greeter-fix`）

**不做**

- PR 发布（`git push` + GitHub API 开 PR）
- SWE-bench Pro 跑批与外部基线对比
- Terminal-Bench 2.0、自建 holdout、多模型扫描
- textual 三栏 TUI
- MCP StreamableHTTP transport（见风险 2）

## 架构

`plan -> act -> observe -> recover` 四段：

- **Plan**：维护 TodoWrite 风格的状态对象，模型每轮整体重写
- **Act**：分派工具调用（读、改、运行、搜索、git）
- **Observe**：捕获 stdout / stderr / 退出码，截断后把摘要喂回
- **Recover**：处理工具错误，既不撑爆上下文，也不无限循环

**Recover**：executor 可用 `report_blocked(reason)` 自报失败（`BlockedReportMiddleware.after_model` 拦截并 `jump_to="end"`，工具不执行）；父图 `recover` 节点复用 `make_plan` 重写剩余 todo，已 `DONE` 项保留在列表头部，触发恢复的失败项就地转为 `DEPRECATED`；上界 `MAX_REPLANS = 3` + 无进展检测（新剩余列表与旧相同 → `recover_no_progress`），超限 → `recover_exhausted`。回退由 planner 决策、executor 用现有 `git` / `edit_file` 执行，不引入快照机制。

**Checkpoint**：父图用 `SqliteSaver` 编译，落 `{worktree}/.agent/checkpoints.sqlite`；`thread_id` 从 `configurable` 读、缺省 `"default"`。自定义类型 `ToDoItem` / `ToDoStatus` 经 `JsonPlusSerializer(allowed_msgpack_modules=...)` 显式放行。**恢复语义 = node 级、at-least-once**：已完成节点不重跑，崩溃时正在执行的节点整段重跑，工具副作用（`edit_file` / `git commit` / 模型调用）不去重，故不提供 exactly-once。`replan_count` 随 state 持久化，resume 不会绕过 `MAX_REPLANS`。**每题必须唯一 `thread_id`**，否则重试会变成续跑。

**Hooks**：`PreToolUse`、`PostToolUse`、`SessionStart`、`SessionEnd`、`UserPromptSubmit`、`Notification`、`Stop`、`PreCompact`——可配置扩展点，运营方在此注入策略、遥测、护栏。

**Stop 落点**：`summary` 节点的 `finally`（`summary.py`）。图内所有终止路径——正常完成 / 预算熔断 / `recover_exhausted` / `recover_no_progress` / 空计划 / 模型异常——都必然写 `{worktree}/.agent/trace.json`。**例外**：交互式 REPL 里被 Ctrl-C 取消的那一轮不走此路径（见「完整交付」）。

**控制台输出落点**：`graph.py` 的 `_announce_todos` 包装 `make_plan` / `start_task` / `end_task` / `recover` 四个会重写 `todo_list` 的节点。打印必须在装配层，不能在 reducer 里：`replace_todos` 每次写入会跑两遍（条件边读一次、`apply_writes` 一次），装配层每个更新只看到一次，结构性免疫。计划版本从 `state["replan_count"]` 取，不放 state 之外的附属属性。容器内 stdout 由 Harbor `tee` 到 `<trial>/agent/langgraph-run.log`，是唯一能看到 agent 实时进度的通道。

**沙箱由 Harbor 提供**：graph 与工具都在容器内执行，宿主文件系统不可达 → spec 的 hard reject「不许在宿主机执行 git」自动满足。

### Harbor 接入契约（0.23.0 源码核实）

| 环节 | 契约 |
| --- | --- |
| 注册 | 项目根 `langgraph.json` → `{"graphs": {"<name>": "./file.py:<attr>"}}`；`<attr>` 可为已编译 graph，或工厂 `make_graph(config)`；Harbor `project_path` 默认为 cwd，须显式指向本项目目录 |
| 调用 | `graph.ainvoke({"messages": [...]}, config={"configurable": {...}})`；state **必须含 `messages`** |
| 返回 | 最终答案取 `result["messages"][-1].content` |
| 模型注入 | `model` / `model_kwargs` / `thread_id` 都在 `configurable` → 从 config 读，**禁止在 import 期固化** |
| worktree | Harbor **不**默认注入 `configurable.worktree`；缺失时回退 `Path.cwd()`（任务工作目录，不是 `/installed-agent/langgraph-project`） |
| token 记账 | runner 自动累加 `result["messages"]` 的 `usage_metadata` |
| 运行位置 | 项目 copytree 进容器 `/installed-agent/langgraph-project`；容器内 venv 为 `/opt/harbor-langgraph-venv`，**Python 3.12**；**不会**带上仓库根 `requirements.txt`，项目须可自安装 |

## 技术栈

- **Python**：本机共享 venv 3.14.6，**容器内 3.12** → 代码须 3.12 兼容
- **编排**：LangGraph + LangChain（model / tool / retriever 抽象）
- **模型**：`langchain-deepseek>=1.1.0`，默认 `deepseek-v4-flash`，provider 与模型名经 `configurable` 注入
- **搜索**：ripgrep 子进程 + tree-sitter（预编译）
- **沙箱 / 评测**：**Harbor 0.23.0**（`uv tool install harbor`，落在 `~/.local/bin`，**默认不在 PATH**）；`--env docker` 为本地默认，`daytona` / `e2b` / `modal` / `runloop` 为云端备选
- **可观测性**：OTel `gen_ai.*` → 本地 `{worktree}/.agent/otel.jsonl`；Langfuse 主路径为 LangChain `CallbackHandler`（`LANGFUSE_PUBLIC_KEY`/`SECRET_KEY`/`BASE_URL`）。可选 `LANGFUSE_OTLP=1` 再挂原始 OTLP（默认关，避免与 Callback 双写）
- **PR 发布**：细粒度 token 的 GitHub App，作用域限目标仓库（MVP 后补）

## 指标与评测

- **主指标**：**手写 Harbor 任务集通过率** = `reward=1` 的任务数 / 任务总数，**逐档报告**（L1–L5 各报通过与否），不报平均值。任务集固定、提交进库，不挑题。
- **任务集**：`harbor_tasks/` 下 **5 个**手写任务，按 **L2 定位难度**逐级递进（L1 可见测试点名函数 → L2 症状在调用方、真因在被调方 → L3 只有现象、无可见测试 → L4 规格散文、可见测试全绿 → L5 架构不变量 + 诱饵）。每题自带 `instruction.md` + verifier（`reward` 0/1）+ 可用的 `solution/solve.sh`；`greeter-fix` 保留为烟测，不计入能力分。任务集是交付物，改它必须同步改本节并说明原因。
- **每题双向可证伪**（本地 `scripts/verify_harbor_tasks_local.sh` 强制）：未修改 → `reward=0`，参考解 → `reward=1`；L2/L5 另加作弊解必须判红。
- **同批记录**：`turns/task`、`tokens/task`（in/out 分开）、`元/task`、`stop_reason`、artifact 收集状态。
- **只与自己比**：不做外部基线对比，数字只用于跨版本比较，不得声称与任何外部 harness 的横向胜负。
- **Harbor 输出目录必须在项目目录之外**（如仓库根 `jobs/`）：`-a langgraph` 会把项目目录整体拷进 trial，输出目录在项目内会自我递归到 `File name too long`。
- **落盘**：`<job>/result.json` + agent 轨迹 + verifier 输出 + `trial.log`（`harbor view <job>` 看轨迹）；汇总进 `eval/results.jsonl`，按任务 id 逐题一行，可跨版本配对。
  - `result.json` 的 `pass_at_k` 实测为 `{}`、`cost_usd` 为 `null`，**不能直接取用**：通过率从 verifier 的 `reward.txt` / `output.json` 自己算，成本从 tokens × 单价自己算（实测 5/5 命中）。
  - **产物收集**：在 `task.toml` 里显式声明 `artifacts = ["/app/.agent/trace.json", "/app/.agent/patch.diff"]` 即修复。实测（2026-10-01 批次）三条 artifact 全部 `status: "ok"`；此前 manifest 全 `empty` 的根因是**任务未声明**，不是 Harbor 的锅。`patch.diff` 另有前置条件：`/app` 必须是 git 仓库（见「已知坑」9）。
- **Patch 落盘**：Stop hook 把 `git status --porcelain` + `git diff HEAD` 写进 `{worktree}/.agent/patch.diff`，同时写进 `/logs/artifacts/patch.diff`。口径限制：只看未提交改动，agent 自己 `git commit` 则 patch 为空（文件头已注明）。

## 预算

| 维度 | 上限 | 熔断行为 |
| --- | --- | --- |
| 轮数 | 150 turns | `Stop` hook → 写 trace |
| 累计 token | 2M tokens | `Stop` hook → 写 trace |
| 成本 | 100/3 元 / task（原 $5，按官网价目折算） | `Stop` hook → 写 trace |

- **turns 口径 = 模型调用次数（LLM invocations）**，不是图循环轮数。planner 调用计 1，executor 每次调用各计 1。
- **累计 token 口径 = 一次 task 内所有模型调用的 `input_tokens + output_tokens` 之和**（含 executor 子调用），**不是**单次上下文窗口占用。单次上下文由 `PreCompact` 单独兜。
- 轮数 150 按实测取值：真实仓库上每 todo ≈ 25 次模型调用，5–6 步计划需 ~150 次。
- `PreCompact` @150k tokens：把旧轮次摘要成 prior-state block，腾出空间但不丢计划。
- 记账来源：`usage_metadata` × 模型单价（人民币，空闲时段）；**`cache_read_tokens` 必须计入**——DeepSeek 有前缀缓存，漏算会让元/task 偏高，且重写历史会令缓存失效。
- **禁止无预算上限运行**：开放式运行会污染跨版本对比。
- **实测**（2026-10-01，6 trial，`-n 2`）：总成本 **0.17 元**、总墙钟 8m29s、每题 17–35 turns / 35k–131k input tokens。原先按「每 todo ≈ 25 次模型调用 × 5–6 todo」估出的 167 元上限**高了约 1000×**——自造 mini-repo 的 todo 数远少于真实仓库，该估算只适用于真实仓库。沙箱容器分钟数与镜像拉取另计，用完即删；本地 docker 近零成本，16GB 内存下 `-n 2` 稳定。

## 交付物

**Harbor 垂直切片（已验证 2026-09-30）**

- 手写任务 `harbor_tasks/greeter-fix` + 项目根 `langgraph.json` + `requirements-harbor.txt`；`harbor run -a langgraph` 在 Docker trial 内调起本项目 `make_graph`，verifier `reward=1`。
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
  - worktree：Harbor 不默认注入 `configurable.worktree`；缺失时 `make_graph` 回退 `Path.cwd()`。

**任务集（2026-10-01 交付，已验证）**

- `harbor_tasks/` 下 5 个能力任务（L1–L5，见「指标与评测」）+ `greeter-fix` 烟测。每题：`instruction.md`、`environment/`（Dockerfile + 可见 repo，`WORKDIR /app` + seed git repo）、`tests/`（隐藏 verifier）、`solution/solve.sh`。
- 本地双向验证：`bash projects/terminal-coding-agent/scripts/verify_harbor_tasks_local.sh` —— 13/13 用例通过（每题 broken→0 / solve→1，L2 与 L5 各含作弊解→0）。用真 Docker 跑，不需要 LLM、不需要 Harbor。
- 跑批命令（cwd = 仓库根；job 落在仓库根 `jobs/`，该目录 gitignore）：
  ```bash
  export PATH="$HOME/.local/bin:$PATH"
  set -a; source local.env; set +a
  harbor run -p projects/terminal-coding-agent/harbor_tasks \
    -a langgraph -m deepseek:deepseek-v4-flash \
    --ae DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY" \
    --allow-agent-host api.deepseek.com -k 1 -n 2 \
    --ak project_path="$(pwd)/projects/terminal-coding-agent"
  ```
  - `-p` 指向父目录**会递归发现全部子任务**（实测 `--dry-run` 报 6 trial），不需要 dataset 清单。
- **实测结果**（job `jobs/2026-10-01__19-28-06`，8m29s，0.17 元）：能力题 **5/5 通过**，逐档 L1=pass L2=pass L3=pass L4=pass L5=pass；6 trial 零异常，`stop_reason` 全 `completed`，artifact 全 `ok`，`patch.diff` 均非空。
  - **本次运行对能力分无区分度**：`deepseek-v4-flash` 把 5 档全解了。阶梯本身可证伪（broken 态确实 0 分），但当前难度点对该模型**已饱和**。下一版要么在同样 5 档内加深（更长的调用链、更隐蔽的诱饵），要么换更弱的模型当被试——两件事都不改任务集的形状。
- 汇总落盘：`python3 projects/terminal-coding-agent/scripts/collect_eval_results.py <job-dir>` → `eval/results.jsonl`（逐题一行：`reward` / `turns` / in-out tokens / `cost_rmb` / `stop_reason` / artifact 状态），并打印逐档报告。

**完整交付（MVP 后补齐）**

- CLI：`terminal-coding-agent`（无子命令，直接进多轮 REPL；`--worktree` / `--session`）
  - 会话语义：单 `thread_id`，`messages` 跨轮累积；`turns/tokens/cost_rmb/todo_list/stop_reason` 等 12 个 per-turn 字段每轮经 `update_state` 重置（预算是 per task）。每轮输入一律作为一个 task 进图，不做 chat/task 路由
  - `--worktree` 缺省为**临时目录**，会话结束即删，`.agent/` 不会落进你的项目；要真让 agent 改某个仓库必须显式传 `--worktree <repo>`，续跑同 `--session` 也需连同传
  - `todo_renderer` / `tool_renderer` 经 `configurable` 注入（`make_graph` → executor）；Harbor 不注入 → todo 走 `_print_todos` 进 `langgraph-run.log`，且不装 `ToolLogMiddleware`。交互式终端下一个 turn 内：todo 面板原地覆盖刷新，与 `working…` 指示器共用一块 `Live`；工具调用一行摘要（`edit_file` 附着色 diff），追加在面板上方，单块原地刷新
  - 每轮页脚只显示 `turns` 与 `stop`。token / cost 明知不准故不展示（provider `usage_metadata` 常缺字段）；记账本身保留，硬闸门仍依赖它
  - **计划闸门（HITL，2026-10-02）**：`configurable["enable_hitl"]` 为真时（只由 CLI 注入），父图插入 `await_plan_approval` 节点，用 LangGraph 动态 `interrupt()` 暂停并把计划摆给用户；`Command(resume="approve")` 继续，`"reject"` → 走 `summary` 写 trace。**两条入口**：`make_plan`（初计划）与 `recover`（replan，`_after_recover_gated`）；replan 那条必须显式接线——`recover` 把 `make_plan` 当**普通函数**调用（`recover.py`），不经图上的边，闸门不在其路径上。**拒绝标签区分两类失败**（评测报告要分类计数）：初计划被否 → `stop_reason="plan_rejected"`；replan 被否 → `"replan_rejected"`，判据是闸门处的 `replan_count`（`recover` 返回前已 +1，先于闸门落地）。**暂停态落 checkpoint**：显式 `--worktree` + 同 `--session` 可在进程被杀后重启重建审批（`pending_interrupts` 读 `get_state().tasks[].interrupts`）。恢复**不**经 `run_task_turn`（不 `update_state`，否则铲平暂停点）。未注入该 flag（Harbor）时节点集与路由与改动前逐字一致
    - **agent 提问（HITL，2026-10-02）**：`enable_hitl` 为真时 executor 多一个 `ask_user` 工具（`middleware/ask_user.py`），模型可在执行途中给人 1–4 个选项并接受自由文本作答。**中断落在 `after_model` 中间件里，即 ToolNode 之前**：LangGraph 的 resume 重放粒度是整个节点，若把 `interrupt()` 放进工具体内，ToolNode 会把同批调用再执行一遍（实测副作用工具执行 **2** 次）；停在 ToolNode 之前则无东西可重放（实测 **1** 次）。这与框架自带的 `HumanInTheLoopMiddleware` 落点一致。`ask_user` 是**信号工具**：`after_model` 就地答掉该调用、把答案注入为带匹配 `tool_call_id` 的 `ToolMessage`，工具 body 永不执行（同 `report_blocked` 先例）。**该调用必须保留在 `tool_calls` 里**——`create_agent` 的 model→边在 `len(tool_calls) == 0` 时直接结束循环，摘掉它会让 resume 后模型再不被调用、答案无人使用（实现期实测抓到的 bug）；保留则路由落到「有调用但无 pending」一支回到模型。契约：选项数不在 1–4 时不中断、回 error `ToolMessage`；一批多个只问第一个；未作答（Ctrl-C / EOF）→ `{"answer": None, "cancelled": True}`，本轮**不终止**，agent 自行判断继续。**不新增 state 字段**（答案走消息流）。未注入该 flag 时工具表与中间件表逐字一致（Harbor 平价）
  - **`executor.py` 必须放行 `GraphBubbleUp`**：`interrupt()` 无论落在工具体内还是中间件里，都会从 `agent.invoke` 抛出；`run_agent` 原有的 `except Exception` 会把它吞成 `stop_reason="executor_error:GraphInterrupt"`，父图**根本不暂停**（实测）。`except GraphBubbleUp: raise` 必须排在 `except Exception` 之前。这是两条 HITL 路径共用的前置条件
- **Ctrl-C 取消（已知降级）**：只中断当前轮并回到提示符，会话继续。`KeyboardInterrupt` 在 REPL 层被捕获，`summary` 不执行 → 该轮不写 `trace.json` / `patch.diff`，checkpoint 停在中途，续跑即从该节点 at-least-once 重放。Harbor 跑批非交互，不受影响。rule `30-agent-architecture` 要求的「取消也要走 Stop hook」**未做**。**两个例外**：审批提示上的 Ctrl-C / EOF 被映射为 `reject`，因此走到 `summary` 并写 trace；提问提示上的 Ctrl-C / EOF 返回「未作答」哨兵，agent 自行判断继续、本轮不终止——是不降级的两处分支
- ≥5 次完整运行的 OTel trace 归档；100% 工具调用带 span
- 报告：任务集的失败分类计数（harness 崩溃 / 预算耗尽 / recover 耗尽 / 模型放弃 / 真改错）；结尾写前三大失败模式与对应 hook 改动
- PR 发布（正文含 plan 与 diff 摘要；**禁止直推 main**）
- `outputs/skill-terminal-coding-agent.md`

## 验收标准（rubric）

| 权重 | 标准 | 度量方式 |
| --- | --- | --- |
| 20 | 架构清晰度 | plan/act/observe 分离、hook 面、工具 schema 可读性 |
| 20 | 安全 | 沙箱逃逸红队 + 破坏性命令守卫审计 |
| 20 | 可观测性 | 100% 工具调用带 span、逐轮 token 记账 |
| 15 | 开发者体验 | 冷启动 < 2s、崩溃恢复、Ctrl-C 取消语义 |

> MVP 覆盖架构 20 + 安全 20 + 可观测性 20；开发者体验 15 属「不做」。
> **满分 75**：spec 的「SWE-bench Pro pass@1」25 分项已放弃，剩余权重不摊回 100。

**硬拒绝项**

- harness 在宿主机文件系统上直接调 git，而非沙箱内执行
- agent 能写出 worktree 之外，或在无显式 allowlist 时 curl 外部 URL
- 上报通过率时更换 / 挑选任务集（必须固定 `harbor_tasks/` 集合并声明版本），或把「基础设施失败」计为通过
- 「通过率」依赖重试之间 `git reset --hard`

## 风险

**未决**

1. **执行器本地还是云端**：MVP 用本地 docker。已实测 Rosetta 下 amd64 仅 **1.39×** 开销（容器内 `vendor_id: VirtualApple`），性能可行；16GB 内存限制并发 ≈ 1，任务集跑批时再定是否上云（只剩「花不花钱」一个变量）。
2. **MCP StreamableHTTP vs 进程内工具**：spec 要求经 MCP 暴露工具，规则要求用 LangChain 做 tool 抽象，两者不同层（wire protocol vs 进程内）。本地同机评测下 MCP 是纯开销，但 rubric 的「工具 schema 可读性」偏向显式 schema。**未决。**
3. **工具面口径**：技术栈栏写 ripgrep + tree-sitter，spec 要求 6 个工具。MVP 按 6 个实现，口径需与实现对齐。

**已知坑**

4. **`strict` 结构化输出的 schema 约束**：DeepSeek `with_structured_output(strict=True)` 要求 schema 中所有 object 属性标 `required`——计划 schema 不能含可选字段；该参数还可能被 API 忽略，当 best-effort 处理，保留校验边界。
5. **`PreCompact` 与消息重建**：摘要 / 裁剪会重写历史；若从纯文本重建消息会丢 `AIMessage.additional_kwargs`，在 thinking 模式触发 400 → 保持原始消息对象，勿重建为 `AIMessage(content=...)`。
6. **模型 ID 时效**：只用 `deepseek-v4-flash` / `deepseek-v4-pro`；`deepseek-chat` / `deepseek-reasoner` 已于 2026-07-24 停用。provider 文档常滞后，勿照抄。
7. **解释器版本错配**：本机 3.14.6 上 `--dry-run` 依赖全部解析通过，**不能**外推为容器 3.12 内可安装；落代码前先在 3.12 下验一次依赖解析。
8. **state 只放可安全往返的值**：list / dict 子类挂自定义属性，经 checkpoint 反序列化后会丢失。需要持久化的标量一律放显式 state 字段。
9. **任务镜像必须 seed 一个 git 仓库**：`patch.diff` 由 Stop hook 跑 `git status --porcelain` + `git diff HEAD` 写出；`/app` 不是 git 仓库时两条命令都失败，patch 恒为空，且 agent 的 `git` 工具全程不可用。Dockerfile 里补 `git init` + 一次 seed commit 即修复（实测 patch 从空变为 17–88 行）。
10. **verifier 默认与 agent 共用容器**（`VerifierEnvironmentMode.SHARED`），所以 test.sh 直接用镜像里的 `pytest`，不走模板的 `uvx` 联网下载——离线、确定，且与 agent 本地跑的是同一版本。代价：agent 若把镜像里的 pytest 弄坏，reward 记 0。
11. **非 UTF-8 字节**：`run_shell` 遇非 UTF-8 输出抛 `UnicodeDecodeError`，会让含二进制数据的任务**无条件 0 分**。本题集全 ASCII 规避；该缺陷应作独立回归任务测，不进能力阶梯。
12. **`docker compose cp failed; retrying …` 是良性的**：Harbor 0.23.0 的 compose cp 与本机 Docker 版本不合，`job.log` 里每次上传/下载都先报一次失败再回退 tar stream / engine cp。文件全部落地（实测 manifest `status: ok`、trace / patch / summary 均在），不影响 reward，不要当故障追。
