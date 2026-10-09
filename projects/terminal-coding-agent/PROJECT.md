# terminal-coding-agent

- **项目**：terminal-coding-agent / 所属 Phase：Phase19 / Capstone 01
- **spec**：[Capstone 01 — Terminal-native coding agent](https://aiengineeringfromscratch.com/lesson?path=phases%2F19-capstone-projects%2F01-terminal-native-coding-agent)
- **状态**：**交付重点在 CLI**。CLI 已交付（多轮 REPL / 输入前缀 / HITL / 8 hook / OTel）；Harbor 评测链路也已跑通（垂直切片 + 5 题任务集，实测 5/5）。**已知降级** 3 处：Ctrl-C 取消的那一轮不写 trace、`!` 无破坏性守卫、`@` 图片不剥离。**未做**：PR 发布、OTel trace 归档、失败报告。

## 目标与范围

**目标**：端到端做出一个 Coding Agent——输入是 CLI，输出是一份 patch——在隔离 worktree 内读改代码、跑命令、自我验证。

**产品是终端 CLI**（`terminal-coding-agent`，多轮 REPL）。**Harbor 是度量它的 harness，不是产品**：spec 要求本项目能被 `harbor run -a langgraph` 在沙箱容器内调起，对 **1 个**手写 Harbor 任务产出 patch 并通过其自带 verifier。为满足这条，项目须保持标准 LangGraph 形状：

- `langgraph.json` + `make_graph(config)` 工厂 + 含 `messages` 的 state schema
- plan / act / observe / recover 四段循环
- **工具（Harbor 侧 7 个）**：`read_file`、`edit_file`（带 diff 预览）、`ripgrep`、`tree_sitter_symbols`、`run_shell`（带 timeout）、`git`，输出一律截断至 4k tokens；第 7 个 `current_time` 无参数、无条件注册。另有第 8 个 `web_search`，由 `configurable["enable_web_search"]` 开启、**默认关**，不计入 Harbor
- 另有 3 个可选调试工具 `debug_start` / `debug_cmd` / `debug_stop`（lldb 持久会话），由 `configurable["enable_debug"]` 开启、**默认关** → Harbor 侧仍 7 个
- **hook 8 个**（spec 要求 ≥4）：`PreToolUse`（破坏性命令守卫）、`PostToolUse`（token 记账）、`SessionStart`（预算初始化）、`SessionEnd`、`UserPromptSubmit`、`Notification`、`Stop`（写 trace）、`PreCompact`
- 三层预算熔断 + `gen_ai.*` OTel span

## 架构

`plan -> act -> observe -> recover`：**Plan** 维护 TodoWrite 风格状态（模型每轮整体重写）· **Act** 分派工具调用 · **Observe** 捕获 stdout / stderr / 退出码，截断后回填 · **Recover** 把失败当 Observation。

**Recover**：executor 可用 `report_blocked(reason)` 自报失败（`BlockedReportMiddleware.after_model` 拦截并 `jump_to="end"`，工具不执行）。父图 `recover` 节点复用 `make_plan` 重写剩余 todo，已 `DONE` 项保留在头部、触发恢复的失败项就地转 `DEPRECATED`。上界 `MAX_REPLANS = 3` + 无进展检测（新剩余列表与旧相同 → `recover_no_progress`），超限 → `recover_exhausted`。回退由 planner 决策、executor 用现有 `git` / `edit_file` 执行，**不引入快照机制**。

**每步上下文**：每步 executor 只拿到 `HumanMessage(该步描述)`，**不带父图 messages 历史**。`run_agent` 把 `SYSTEM_PROMPTS["executor"]` + `format_todos(全量 todo_list)`（当前步标 `[+]`）+ `Your step: <当前步>` 拼成该步 system prompt。**计划是步间唯一共享契约**；依赖运行时才产生的字面量看不到（有意）。

**Checkpoint**：父图用 `SqliteSaver` 编译，落 `{worktree}/.agent/checkpoints.sqlite`；`thread_id` 从 `configurable` 读、缺省 `"default"`，**每题必须唯一**（否则重试变成续跑）。`ToDoItem` / `ToDoStatus` 经 `JsonPlusSerializer(allowed_msgpack_modules=...)` 放行。**恢复语义 = node 级 at-least-once**：已完成节点不重跑，崩溃时正在执行的节点整段重跑，工具副作用不去重，**不提供 exactly-once**。`replan_count` 随 state 持久化，resume 绕过不了 `MAX_REPLANS`。

**Stop 落点**：`summary` 节点的 `finally`（`summary.py`）。图内所有终止路径——正常完成 / 预算熔断 / `recover_exhausted` / `recover_no_progress` / 空计划 / 模型异常——都必然写 `{worktree}/.agent/trace.json`；**唯一例外**是 REPL 里被 Ctrl-C 取消的那一轮。

**intent 门（chat / work 分流）**：CLI 注入 `configurable["enable_intent"]=True` 时，图首多一个 `intent` 节点（共享组件 `intent`，`gate.build_intent` 装配，模型同 planner，`work_hint` = 「需计划并在 worktree 里动手」）。它一次结构化调用判该轮是 `chat` 还是 `work`：`chat` → 新节点 `chat`（一次 executor 调用、带工具，system prompt 要求内容进回复而非写文件），末条 AI 消息写入 `messages`，`summary` 见 `intent=chat` 且末条是非空 AI 消息即**原样转达**、不调模型（产出为空则降级 summarize）；`work` → 原 `make_plan` 管线。组件在 `gate.build_intent` 内**懒 import**，Harbor 导入链不依赖它；判失败按「拿不准一律 work」降级并记日志。判据是「是否必须进主流水线」（C 轴），故「写一篇作文」「看一眼 README」属 chat，判据与工作区是否改动无关。分类器**只看最近 3 条有人类文本的 turn**（工具消息 / 纯工具调用的 assistant / 图片 part 都不占额度，单条截 200 字符加 `...`）；整形在组件内完成，故 `tool_calls` 到不了结构化调用——此前喂整份 `messages` 时，模型会照抄历史里的工具名而输出失败（约 1/20），被本节点的 `except` 吞成 `work`、令那些轮的 chat 支静默失效。详见 `components/docs/features/intent-context/design.md`。

**CLI 与 Harbor 的单一图**：CLI 的三个 flag（`enable_intent` / `enable_hitl` / `enable_web_search`）都只从 `configurable` 注入，图上不预置分支。不注入时（即 Harbor 路径）planner schema / planner 调用 / 节点集 / 路由 / 工具表保持原状（schema 恒为 `Plan`；`SYSTEM_PROMPTS["planner"]` 只在 `enable_intent` 打开时注入；无 `intent` / `chat` 节点、无闸门、工具表 7 个）——**同一份图既能被 Harbor 跑批，也能被 CLI 交互使用**，不存在两套实现。

**控制台输出**：`graph.py` 的 `_announce_todos` 包装 `make_plan` / `start_task` / `end_task` / `recover` 四个重写 `todo_list` 的节点。打印**必须在装配层**——`replace_todos` 每次写入跑两遍（条件边 + `apply_writes`），装配层每个更新只看到一次。计划版本取自 `state["replan_count"]`。容器内 stdout 由 Harbor `tee` 到 `<trial>/agent/langgraph-run.log`，是跑批时唯一能看到实时进度的通道。

**沙箱**：Harbor 侧 graph 与工具都在容器内执行，宿主文件系统不可达 → spec 的 hard reject「不许在宿主机执行 git」自动满足。

**模块路径**：`graph.py`（装配）· `cli.py`（REPL 入口）· `attachments.py`（`strip_images` / `attached_images`，不 import `repl_console`）· `tools/`（唯一副作用边界）· `debug_worker.py`（lldb 会话 worker，独立 Python 3.9 进程）· `ui.py` · `middleware/` · `harbor_tasks/` · `scripts/`（双向验证 + 结果汇总）· `tests/`。

**图产物**：`graph.png` 由 `python -m terminal_coding_agent.graph` 生成，默认**开着两个闸门**（`intent` + HITL）以展示 CLI 拓扑；`ENABLE_HITL=0` / `ENABLE_INTENT=0` 可渲染 Harbor 的拓扑。拓扑改了重跑；`example.png` 为手工截图。README 只放这两张。

## 技术栈

- **Python**：本机共享 venv 3.14.6，**容器内 3.12** → 代码须 3.12 兼容
- **编排**：LangGraph + LangChain（model / tool / retriever 抽象）
- **调试**：lldb Python API（随 Xcode Command Line Tools），绑定只支持 Python 3.9 → 会话跑在独立 `python3.9` worker 子进程，经逐行 JSON 协议驱动；**不解析 lldb console**。默认关。
- **模型**：`langchain-deepseek>=1.1.0`，默认 `deepseek-v4-flash`；provider 与模型名经 `configurable` 注入
- **搜索**：ripgrep 子进程 + tree-sitter（预编译）；可选 `ddgs`（DuckDuckGo 文本检索，无 API key，仅本地 CLI 开）
- **可观测性**：OTel `gen_ai.*` → `{worktree}/.agent/otel.jsonl`；Langfuse 主路径为 LangChain `CallbackHandler`（`LANGFUSE_PUBLIC_KEY` / `SECRET_KEY` / `BASE_URL`）；可选 `LANGFUSE_OTLP=1` 挂原始 OTLP（默认关，避免双写）
- **沙箱 / 评测**（非交付重点，仅度量用）：**Harbor 0.23.0**（`uv tool install harbor`，落 `~/.local/bin`，**默认不在 PATH**）；`--env docker` 为本地默认，`daytona` / `e2b` / `modal` / `runloop` 为云端备选

## 交付物

**CLI 与运行时（交付重点，已交付）**：`terminal-coding-agent` 无子命令，直接进多轮 REPL（`--worktree` / `--session`）。用户可见用法见 `README.md`。四条要记住的：

- 单 `thread_id`、`messages` 跨轮累积；12 个 per-turn 字段每轮经 `update_state` 重置（预算是 per task）；**恢复暂停点时不走 `run_task_turn`**，否则会铲平它。
- `--worktree` 缺省为**临时目录**，会话结束即删；要真让 agent 改某个仓库必须显式传 `--worktree <repo>`，续跑同 `--session` 也需连同传。
- **输入前缀** `@path`（附文件 / 图片，只给 executor）· `!cmd`（本地 shell，不进上下文、不计预算）· `/cmd`（REPL 命令表）。三者由共享组件 `repl-console` 提供（见根 `README.md` 组件表）。
- `todo_renderer` / `tool_renderer` 经 `configurable` 注入；Harbor 不注入 → todo 走 `_print_todos` 进 `langgraph-run.log`。**Harbor 的图导入链不依赖 `repl_console` / `prompt_toolkit`**，`requirements-harbor.txt` 不增加它们。

**lldb 调试工具（2026-10-09，已交付）**：`debug_start` / `debug_cmd` / `debug_stop` 三件套，agent 在一个活会话里连续设断点、运行、看变量与调用栈。会话住独立 `python3.9` worker（绑定限 3.9），主进程经逐行 JSON 驱动，**不解析 console**；超时由 watchdog 线程 `SBProcess.Stop()` 打断、会话仍可用。由 `enable_debug` 开启、默认关 → Harbor 工具表逐字不变。**只在本地 CLI 交付**（沙箱无 lldb）。

**Harbor 评测证据（已交付，非当前重点）**：`harbor_tasks/` 下 5 题能力集（`l1`–`l5`，按 L2 定位难度递进）+ `greeter-fix` 烟测，配 `langgraph.json` + `requirements-harbor.txt`。

- **跑批**（cwd = 仓库根，job 落仓库根 `jobs/`）：

```bash
export PATH="$HOME/.local/bin:$PATH"
set -a; source local.env; set +a
harbor run -p projects/terminal-coding-agent/harbor_tasks \
  -a langgraph -m deepseek:deepseek-v4-flash \
  --ae DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY" \
  --allow-agent-host api.deepseek.com -k 1 -n 2 \
  --ak project_path="$(pwd)/projects/terminal-coding-agent"
```

- **模型与凭据可换**：`configurable["planner"]` / `["executor"]` 收 `provider:model` 串（缺省 `deepseek:deepseek-v4-flash`），`configurable["local_model"]` 走本地 Ollama（不取 API key）；上面命令里的 `-m` 与 `--ae` 随所选 provider 一起改。
- **实测**（job `jobs/<跑批目录>`，8m29s，0.17 元）：能力题 **5/5**，逐档 L1–L5 全 pass；6 trial 零异常，`stop_reason` 全 `completed`，artifact 全 `ok`，`patch.diff` 均非空。本地双向验证 **13/13**（broken→0 / solve→1，L2 与 L5 各含作弊解→0）。
- **当前无区分度**：`deepseek-v4-flash` 把 5 档全解了。阶梯可证伪（broken 态确实 0 分）但已饱和；下一版在同样 5 档内加深，或换更弱模型当被试。
- 汇总：`python3 scripts/collect_eval_results.py <job-dir>` → `eval/results.jsonl`（逐题 `reward` / `turns` / in-out tokens / `cost_rmb` / `stop_reason` / artifact 状态）。

**已知降级**：REPL 里被 Ctrl-C 取消的那一轮不写 `trace.json` / `patch.diff`，checkpoint 停在中途，续跑即从该节点 at-least-once 重放；Harbor 跑批非交互，不受影响。rule `30-agent-architecture` 要求的「取消也要走 Stop hook」**未做**。（`!` 无破坏性命令守卫、`@` 图片不剥离，属首版刻意取舍。）

