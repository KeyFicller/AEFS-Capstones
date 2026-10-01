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

- 多轮 CLI（`chat` REPL）与 rich 内联 UI —— 已实现，见「交付物」；textual 三栏 TUI 仍不做
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

- **Recover 落地**：executor 可用 `report_blocked(reason)` 自报失败（`BlockedReportMiddleware.after_model` 拦截并 `jump_to="end"`，工具不执行）；父图 `recover` 节点复用 `make_plan` 重写剩余 todo，已 `DONE` 项保留在列表头部，触发本次恢复的失败项就地转为 `DEPRECATED`（不再算失败、不重跑）；上界 `MAX_REPLANS = 3` + 无进展检测（planner 新剩余列表与旧相同 → `recover_no_progress`），超限 → `recover_exhausted`。回退由 planner 决策、executor 用现有 `git`/`edit_file` 执行，不引入快照机制。

- **Checkpoint 落地**：父图用 `SqliteSaver` 编译，落 `{worktree}/.agent/checkpoints.sqlite`；`thread_id` 从 `configurable` 读、缺省 `"default"`（调用时传入可覆盖）。自定义类型 `ToDoItem` / `ToDoStatus` 经 `JsonPlusSerializer(allowed_msgpack_modules=...)` 显式放行。**恢复语义 = node 级、at-least-once**：已完成的节点不重跑，崩溃时正在执行的节点整段重跑，工具副作用（`edit_file` / `git commit` / 模型调用）**不去重**，故**不提供 exactly-once**。`replan_count` 在 state 内被持久化，resume 不会绕过 `MAX_REPLANS`。**每题必须唯一 `thread_id`**，否则重试会变成续跑（等同硬拒绝项「重试间刷分」）。

**Hooks**：`PreToolUse`、`PostToolUse`、`SessionStart`、`SessionEnd`、`UserPromptSubmit`、`Notification`、`Stop`、`PreCompact`——可配置扩展点，运营方在此注入策略、遥测、护栏。

- **Stop 落点**：`summary` 节点的 `finally`（`summary.py`）。任何终止路径——正常完成 / 预算熔断 / `recover_exhausted` / `recover_no_progress` / 空计划 / 模型异常——都必然写 `{worktree}/.agent/trace.json`，不存在漏写分支。

- **控制台输出落点**：`graph.py` 的 `_announce_todos` 包装 `make_plan` / `start_task` / `end_task` / `recover` 四个会重写 `todo_list` 的节点。**打印必须在装配层，不能在 reducer 里**：`replace_todos` 每次写入会跑两遍（条件边读一次、`apply_writes` 一次），旧实现为此不得不加模块级 `_last_printed` 去重；装配层每个更新只看到一次，结构性免疫。计划版本直接从 `state["replan_count"]` 取，故不需要 `_VersionedTodos` 那个 list 子类（风险 8）。容器内 stdout 由 Harbor `tee` 到 `<trial>/agent/langgraph-run.log`，是唯一能看到 agent 实时进度的通道。

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
- **可观测性**：OTel `gen_ai.*` → 本地 `{worktree}/.agent/otel.jsonl`；Langfuse 主路径为 LangChain `CallbackHandler`（`LANGFUSE_PUBLIC_KEY`/`SECRET_KEY`/`BASE_URL`）。可选 `LANGFUSE_OTLP=1` 再挂原始 OTLP（默认关，避免与 Callback 双写）。
- **PR 发布**：细粒度 token 的 GitHub App，作用域限目标仓库（MVP 后补）

## 指标与基线

- **主指标**：`pass@1`（30 题中首次运行即通过的比例）。**禁止在重试之间 `git reset --hard` 刷分**。同批记录 `turns/task`、`tokens/task`（in/out 分开）、`元/task`。
- **基线**：`mini-swe-agent`，**必须同模型、同 30 题**；模型不同则测的是模型差距而非 harness 差距，对比作废。Live-SWE-agent 仅作上下文参照，不参与打分。
- **数据集**：SWE-bench Pro V2 的 python 子集 **30 题**（`repo_language == "python"`，取自每题 `tests/config.json`）；抽样规则固定（instance_id 列表 + seed）落 `eval/tasks.json`，否则 matched subset 不成立。数据集用 Harbor 注册表 `scale-ai/swe-bench-pro`（731 题），每题镜像由任务自带 `environment/Dockerfile` 的 `FROM` 决定，形如 `jefzda/sweap-images:<tag>`（Docker Hub，**linux/amd64**）。
- **镜像可达性**：本机 `auth.docker.io` 被 DNS 污染、拉不动。必须先经镜像源预拉再重打 tag：
  `docker pull --platform linux/amd64 dockerproxy.net/jefzda/sweap-images:<tag>` → `docker tag dockerproxy.net/... jefzda/sweap-images:<tag>`。
- **Harbor 输出目录必须在项目目录之外**（如仓库根 `jobs/`）。`-a langgraph` 会把项目目录整体拷进 trial，输出目录若在项目内会自我递归到 `File name too long`。
- **落盘**：Harbor 原生输出 `pass_at_k` / `cost_usd` / token 统计到 `<job>/result.json`，另有 agent 轨迹、verifier 输出、`trial.log`（`harbor view <job>` 看轨迹）；汇总进 `eval/results.jsonl`，基线同 schema 单独落盘以便逐题配对。
- **Patch 落盘**：Stop hook（`summary.py` 的 `finally`）把 `git status --porcelain` + `git diff HEAD` 写进 `{worktree}/.agent/patch.diff`，同时写进 Harbor 约定目录 `/logs/artifacts/patch.diff`。docker 后端下该目录是**宿主挂载**，文件直接出现在 `<job>/<trial>/artifacts/logs/artifacts/patch.diff`（manifest 的 `status` 从 `empty` 变 `ok` 即表示已收集，无需额外注册 artifact）。**口径限制**：只看未提交改动——`git diff` 用 `HEAD` 是为了覆盖 staged，若 agent 自己 `git commit` 则 patch 为空（文件头已注明，不假装「无改动」）。

## 预算

| 维度 | 上限 | 熔断行为 |
| --- | --- | --- |
| 轮数 | 150 turns | `Stop` hook → 写 trace |
| 累计 token | 2M tokens | `Stop` hook → 写 trace |
| 成本 | 100/3 元 / task（原 $5，按官网价目折算） | `Stop` hook → 写 trace |

- **turns 口径 = 模型调用次数（LLM invocations）**，不是图循环轮数。planner 调用计 1，executor 每次调用各计 1。选此口径的唯一理由：`mini-swe-agent` 的 step 计数同义，基线对比的 `turns/task` 才可比。
- **累计 token 口径 = 一次 task 内所有模型调用的 `input_tokens + output_tokens` 之和**（含 executor 子调用），**不是**单次上下文窗口占用。单次上下文由下方 `PreCompact` 单独兜。实测真实仓库上约 13k tokens/次调用，故 2M 与 150 turns 大致同档，先到先触发；作用只是拦住「单次调用把上下文炸成超大 prompt」这种病态情况，真正的常驻闸门仍是轮数。
- **轮数取值依据**：2026-10-01 在 `openlibrary-e8084193` 上实测，每 todo ≈ 25 次模型调用，5-6 步计划需 ~150 次；旧值 50 只够 2 个 todo，agent 定位到 `read_publisher` 却无预算落编辑。成本此时仅 0.108 元，离 33 元上限两个数量级。

- `PreCompact` @150k tokens：把旧轮次摘要成 prior-state block，腾出空间但不丢计划。
- 记账来源：`usage_metadata` × 模型单价（人民币，空闲时段）；**`cache_read_tokens` 必须计入**——DeepSeek 有前缀缓存，漏算会让元/task 偏高，且重写历史会令缓存失效。
- **禁止无预算上限运行**：开放式运行会污染评测对比。
- 30 题 × 100/3 元 = **1000 元上限**（仅模型成本）。沙箱容器分钟数与镜像拉取（每题数 GB）**另计**，用完即删；本地 docker 近零成本，但 16GB 内存把并发压到约 1。

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

- CLI：`terminal-coding-agent`（无子命令，直接进多轮 REPL；`--worktree` / `--session`）
  - 会话语义：单 `thread_id`，`messages` 跨轮累积；`turns/tokens/cost_rmb/todo_list/stop_reason` 等
    12 个 per-turn 字段每轮经 `update_state` 重置（PROJECT.md 的预算是 per task）
  - `todo_renderer` 经 `configurable` 注入；Harbor 不注入 → 走 `_print_todos` 进 `langgraph-run.log`。
    交互式终端下一个 turn 内：todo 面板**原地覆盖**刷新（标题 `Tasks`，subtitle 始终为 `replan vN`），
    且共用一块 `Live` 显示自转的 `working…` 指示器（非交互式 / 区域外仍逐态打印）
  - `tool_renderer` 同类注入（`configurable` → `make_graph` → executor），由 `ToolLogMiddleware` 在
    **工具真正执行后**回调（`SafetyMiddleware` 拦截的调用不产生输出）；Harbor 不注入则不装该中间件。
    每次调用一行摘要（工具名 + 关键参数 + 结果摘要），`edit_file` 额外打印着色 unified diff
    （`background_color="default"` 避免默认主题把行补齐到 80 列、在窄终端折行）。渲染与 `TodoPanel`
    共用同一 `Console`，故工具输出**追加在上、Tasks 面板钉在最底**且仍单块原地刷新
  - 每轮页脚只显示 `turns` 与 `stop`。**token / cost 明知不准故不展示**（provider `usage_metadata`
    常缺字段，展示即误导）；记账本身保留 —— `MAX_TOKENS` / `MAX_COST_RMB` 硬闸门仍依赖它
  - `--worktree` 缺省为**临时目录**，会话结束即删；`.agent/`（checkpoint / trace / otel / patch）
    因此不会落进你的项目。要真让 agent 改某个仓库，必须显式 `--worktree <repo>`；
    续跑同 `--session` 也需连同 `--worktree` 一起传
  - 每轮输入一律作为一个 task 进图（不做 chat/task 路由）
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
8. **list 子类挂属性不能进 state**：`_VersionedTodos` 这类「list 子类 + 自定义属性」的 state 在 checkpoint 反序列化后属性会**丢失**（实测 `replan_version` 读回为 `<LOST>`）。state 只放能被 msgpack（含白名单）安全往返的值；replan 版本改用已有的 `replan_count` 渲染。
