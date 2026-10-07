# cyber-cricket

- **项目**：cyber-cricket（游戏无关的 bot 竞技平台）/ 所属 Phase：Phase19 / Capstone 10（**改写**：交付物从「一个 PR」换成「一个 bot」）+ Phase 16
- **spec 链接**：[Capstone 10 — Multi-Agent Software Engineering Team](https://aiengineeringfromscratch.com/lesson?path=phases%2F19-capstone-projects%2F10-multi-agent-software-team) · [Phase 16 — Multi-Agent & Swarms](https://aiengineeringfromscratch.com/lesson?path=phases%2F16-multi-agent-and-swarms%2F01-why-multi-agent)
- **设计**：`docs/features/arena-core/design.md`
- **状态**：立项版。设计已出，**尚未落地**；落地后回写本文件。

## 目标与范围

**目标**：做出「赛博斗蛐蛐」的地基——一个**游戏无关的 bot 竞技平台**：多家（未来的 LLM 多智能体研发队）各自写一个 bot，同一款游戏上互斗。平台提供**可插拔的游戏规则**、**评分 / 排名**、**基准 bot**。Capstone 10 的多智能体研发队形状不改，只把交付物从 PR 换成 bot；Phase 16 的协调机制（角色分工、debate、投票、共识、声誉、失败模式）留给研发队那一层。

**MVP（第一版只做这些）**——见 design 第 1.1 节

- C++20 引擎核心 + CMake/CTest；构建与测试复用根 `./start.sh run`，**不改 `start.sh`**
- 游戏插件接口（编译期注册的 C++ 抽象接口）
- **bot ABI v1（稳定 C ABI）+ `dlopen` 动态加载**：离散 `int32` 动作、游戏自定义 POD `view`，零序列化 / 零 IPC / 无 JSON
- **2 款 2–N 人自由混战游戏**：炸弹人（同时回合 / 有 RNG 布局 / 按存活排名）、多人抢线（顺序回合 / 无 RNG / 先成线即胜）
- 基准 bot：平台自带通用 `random`（零游戏知识）+ 每款游戏各一个 `greedy`
- 循环赛 + Elo（bootstrap CI + 名次翻转率）→ `results.tsv` + `leaderboard.md`
- 可复现回放：种子 + 动作序列 → 断言末态 `digest` 一致
- CLI 观战（ASCII 实时盘面）+ `bench`（matches/sec）

**不做**

- **LLM 多智能体研发队自动写 bot**——下一个 feature；本轮只保证接缝（bot 是编译产物 / roster 多参赛者 / 每回合延迟可记账）
- 非 C-ABI 语言的 bot（Python / JS 需桥接或进程 transport）
- 游戏插件也走 `dlopen`；网络 / 分布式对局；bot 市场 / 声誉经济；Web 观战 UI；自对弈 RL / MCTS
- 隐藏信息游戏（协议已留 per-seat `view` 的口，MVP 两款都是完全信息）

## 架构

见 design 第 2–7 节。一句话：引擎是仲裁者，**游戏不知道 bot 存在**；`Game` 接口只有 11 个方法（`initial / actors / view / legal / apply / terminal / render / digest` …），RNG 只能来自引擎；bot 是 `dlopen` 进来的动态库，只导出 4 个 C 符号（`info / init / act / end`）；对局在 `fork` 出的 **worker 池**里跑（bot 库在 worker 内跨局复用），父进程按整局墙钟监督、崩溃即重启——于是同时拿到「零 IPC 的速度」和「崩溃 / 死循环只丢当前局」。

模块边界、两款游戏的规则口径（含炸弹人每步结算次序）、回放格式见 design 第 2.1 / 5 / 7 节。

## 技术栈

- **C++20**，Apple clang 17，CMake 4.3.2，CTest。本机 10 核、无 `ninja`（用 Makefiles）。
- **零第三方依赖**：不用 JSON 库（接口是 POD + `int32`）、不用第三方测试框架（CTest 自带）、哈希手写 FNV-1a。

**与仓库共享规则的冲突声明**：本项目是 Python 单栈仓库里的 **C++ 例外**。按 `.cursor/rules/workflow.mdc`「PROJECT.md 是项目特有信息的唯一来源，与规则冲突以它为准」，本项目覆盖 `.cursor/rules/python.mdc` 的选型条款：

| python.mdc 条款 | 本项目做法 | 理由 |
| --- | --- | --- |
| 编排用 LangGraph、模型经 LangChain | 不适用（无模型调用） | 本 feature 是竞技平台，不是 agent |
| 依赖唯一来源根 `requirements.txt` | 不适用（零第三方依赖） | C++ 侧无 pip 依赖 |
| 子项目以 `-e ./projects/<p>` 加入共享 venv | **不写 `pyproject.toml`、不改 `requirements.txt`** | 本项目不是 Python 包 |
| 环境 / 测试统一走 `./start.sh` | **复用 `./start.sh run`**，不改 `start.sh` | 构建 / 测试是 cmake / ctest，`run` 已能透传 |

`.gitignore` 需加一行 `projects/*/build/`。

## 指标与评测

主指标见 design 第 10 节：抽象成立（2 款游戏共用 1 个 `Game` 接口，`rg 'game ==' src/core` 零命中）、确定性（同种子 digest 相同 + 回放一致）、鲁棒（四类故障 bot 有确定结果且无挂起）、吞吐（`bench` matches/sec）、排名（`greedy` 相对 `random` 以 CI 显著胜出）、观战。

可证伪假设：**H1** 抢线（7×7、4 人）单核 ≥ 10,000 局/秒；**H2** 同种子 tournament 连跑两次 `ranks` 逐行相同；**H3** `greedy` vs `random` ≥ 200 局 Elo 差 > 200 且 95% CI 不跨 1500。

**实测**：尚未落地，无数字可报。

**硬拒绝项**：硬编码 / 预置比赛结果；`bench` 或指标走旁路；任何非引擎随机源；回放不真重放；挑 roster / 种子 / 重复次数上报排名；把 `load_error` 计为胜场或平局；声称与任何外部 bot 竞技平台可比。

## 预算

**无模型调用**，预算是算力口径（design 第 11 节）：单步 `deadline_ms`（默认 20 ms，**协作式**）→ 单局墙钟（默认 5 s，父进程强制）→ 单局步数上限（游戏参数）→ 循环赛 `--max-seconds`（可选）。

## 交付物

**尚未落地。** 计划：

- CLI `cyber-cricket`：`games` / `play` / `tournament` / `replay` / `bench`
- 引擎：`src/core/`（rng、game、match、bot、worker、elo、tournament、replay、tsv、registry）
- 游戏：`src/games/bomber/`、`src/games/line/`（各含规则 + `view.h` + `render` + `greedy` bot 库）
- 基准 bot：`src/bots/random/`（通用）+ 各游戏一个 `greedy`
- 测试：`tests/`（CTest：抽象无特判、确定性、回放、四类故障 bot）
- 评测：`eval/results.tsv`、`eval/leaderboard.md`、`bench` 数字
- `README.md`：四段（标题 + 一句话 / ASCII 观战截图 / 抽象框图 / 跑起来）。**本项目没有 LangGraph，故 `readme.mdc` 要求的 `graph.png` 用抽象框图替代**——落地时在此写清替代方式，不假装有 LangGraph 图

## 验收标准（rubric）

沿用 Capstone 10 的五个维度，但按「平台」而不是「一次 SWE 任务」计分：

| 权重 | 标准 | 度量方式 |
| --- | --- | --- |
| 25 | 抽象成立 | 2 款结构不同的游戏共用 1 个 `Game` 接口；核心零 `game ==` 特判 |
| 20 | 确定性与可复现 | 同种子 digest 相同；回放断言通过；H2 |
| 20 | 鲁棒与隔离 | 四类故障 bot 各有确定结果且无挂起；worker 崩溃只丢当前局 |
| 20 | 评分 / 排名质量 | `greedy` vs `random` CI 显著；名次翻转率随 repeats 下降；H3 |
| 15 | 观战与开发体验 | ASCII 实时盘面；CLI 五子命令可用；`bench` 出数（H1） |

> 与 Capstone 10 的差异：spec 的「SWE-bench Pro pass@1 / 并行加速 / reviewer 误放行」三项对应到**「平台抽象 / 吞吐 / 排名质量」**——本 feature 不含 LLM 评审，那三项属研发队那一层（下一个 feature）。

**硬拒绝项**：见「指标与评测」。

## 风险

见 design 第 12 节。落地前最该盯的三条：

1. **炸弹人的结算次序 + 参数平衡**（`F`、`R`、网格尺寸）：次序已在 design 2.1 钉死，但参数是否「有得打」需实测（`greedy` vs `random`）后再定。
2. **抢线在 4 人下的先手优势**：座位轮换缓解但不消除；实测偏差要如实报告，不调参掩盖。
3. **C ABI 的编译期契约**：改 `ArenaTurn` 布局必须升 `abi` 并拒绝旧库；`view_size` 随参数变化，bot 必须校验后再 `reinterpret_cast`。
