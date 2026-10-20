# 难任务集（h5–h7）四臂重测

**跑批**：2026-10-20 · `deepseek-v4-flash` · Harbor 0.23.0 · 3 题（`h5`–`h7`）× `-k 4` = 每臂 12 trial
**网络**：重测前已确认 DeepSeek API 200、Ubuntu 源 200；上一轮 4h 异常归因于网络中断，本轮四臂合计 66 分钟无 infra 错误。
**复现**：`bash projects/terminal-coding-agent/scripts/run_hard_arms.sh <arm> [configurable-json]`

## 本地可解性（无 LLM）

`bash scripts/verify_harbor_tasks_local.sh` → **21/21 checks OK**，含旧四题无回归。

| 任务 | broken | solve | cheat | 破损时可见测试 |
|---|---|---|---|---|
| h5-stock-holds | 0 ✓ | 1 ✓ | 0 ✓ | **绿** |
| h6-ledger-dates | 0 ✓ | 1 ✓ | 0 ✓ | 红（1 条） |
| h7-event-rollup | 0 ✓ | 1 ✓ | 0 ✓ | **绿** |

## 四臂汇总

| 臂 | 通过 | pass@4 | 平均轮次 | 最大轮次 | 输入 tok | 输出 tok | 耗时 |
|---|---|---|---|---|---|---|---|
| executor_only | 11/12 | 1.0 | 20.1 | 39 | 223,587 | 11,954 | 17m13s |
| executor_only + shell | 9/12 | 1.0 | 26.7 | 151 | 190,659 | 8,966 | 21m08s |
| native_todo | **12/12** | 1.0 | 33.9 | 151 | 227,580 | 13,764 | 10m55s |
| native_todo + shell | 11/12 | 1.0 | 39.1 | 151 | 192,636 | 13,576 | 16m51s |

合计 43/48。

### 逐题

| 任务 | executor_only | +shell | native_todo | +shell |
|---|---|---|---|---|
| h5-stock-holds | 4/4 | **3/4** | 4/4 | 4/4 |
| h6-ledger-dates | 4/4 | **3/4** | 4/4 | 4/4 |
| h7-event-rollup | **3/4** | **3/4** | 4/4 | **3/4** |

## 失败的三种形态

**这轮没有一次 infra 错误**，5 次未通过全部是 agent 自身行为：

| trial | 轮次 | stop_reason | 说明 |
|---|---|---|---|
| h7 @ executor_only | — | `AgentTimeoutError` | 900s 用尽 |
| h5 @ executor_only+shell | 5 | `blocked` | 只复述任务就停，没动手 |
| h6 @ executor_only+shell | **151** | `max_turns` | 撞模型调用上限，改了 4 个文件仍算错（整月算出 9643 而非 10000） |
| h7 @ executor_only+shell | — | `AgentTimeoutError` | 900s 用尽 |
| h7 @ native_todo+shell | **151** | `max_turns` | 撞上限；`unique_sessions` 算成 3，应为 2 |

### 结论

1. **通过率仍然饱和。** 四臂 `pass@4` 全为 1.0；`n=4` 下 43/48 与 48/48 的差异不显著（Wilson 区间大幅重叠）。**不能**据此排名拓扑。
2. **真正的区分度在「代价」和「卡住」，不在分数。** 平均轮次 20.1 → 26.7 → 33.9 → 39.1，单调上升；三次撞 151 轮上限全部发生在「更重」的配置上。
3. **`max_turns` 是最有价值的信号。** h6 那次 151 轮改了 4 个文件（`settings.py` / `policy.py` / `proration.py` / `dates.py`）却把整月算成 9643——它把两个根因都找到了，然后在重写比例分摊时引入新 bug，并陷入无限调试。这正是「需要跨模块推理」的题目该产生的失败形态。
4. **h5 的 `blocked` 不是难度信号。** 5 轮、仅 967 输出 token 就停，看起来是模型复述任务后未继续，属行为抖动。
5. **`shell_anything` 在两种拓扑下都微降通过率**（12→11、12→11），与 H1–H4 的结论一致：它不是瓶颈，也不提供收益。

## 下一轮要改的（按性价比排序）

1. **`instruction.md` 泄露答案要点。** h5 的 instruction 直接写了「过期回收、幂等键、确认/释放记账」三个方向，等于替模型完成了定位。改成只给用户故事（「客户说库存显示 0 但实际有货」），不给模块名与缺陷清单，才能测到仓库阅读能力。
2. **`agent.timeout_sec` 900 太短，掩盖了失败原因。** 两次 h7 超时都没留下可供判断的中间状态；提到 1800 后重跑，区分「慢」与「卡死」。
3. **`max_turns` 上限 151 是个好旋钮。** 现在撞上限直接判 0。若要量化「卡住程度」，可记录撞上限前的最后 diff，看它是否在收敛。
4. **h7 需要更多正确性用例。** native_todo+shell 那次是 `unique_sessions` 差 1，说明 session 链式传递的边界仍容易被误读；补 4–6 条针对链式合并的定向用例。
5. **若要真正拉开分数**，唯一杠杆是让正确解**必须改调用方**（跨模块签名变更），目前三题都能在被调函数内部修完。

`h1`–`h4` 未改动，旧数据仍可复现。
