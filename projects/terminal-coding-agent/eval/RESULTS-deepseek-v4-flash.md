# 评测结果：原生 TodoListMiddleware vs executor-only

**跑批**：2026-10-19 · `deepseek-v4-flash` · Harbor 0.23.0 · 4 题（`h1`–`h4`）× 2 次尝试 = 每臂 8 trial
**命令模板**（cwd = 仓库根，`-n 4` 并发）：

```bash
harbor run -p projects/terminal-coding-agent/harbor_tasks \
  -a langgraph -m deepseek:deepseek-v4-flash \
  --ae DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY" --allow-agent-host api.deepseek.com \
  -k 2 -n 4 --job-name <job> \
  --ak project_path="$(pwd)/projects/terminal-coding-agent" \
  --ak 'configurable={"executor_only": true}'
```

各臂的 `--ak configurable=...`：

| 臂 | configurable |
|---|---|
| executor_only | `{"executor_only": true}` |
| executor_only + shell-only | `{"executor_only": true, "shell_anything": true}` |
| native_todo | 无（默认） |
| native_todo + shell-only | `{"shell_anything": true}` |

## 汇总

| 臂 | 通过 | Wilson 95% | 平均轮次 | 轮次中位 | 平均输入 tok | 输入中位 | 平均输出 tok | 异常 |
|---|---|---|---|---|---|---|---|---|
| **executor_only**（基准） | 8/8 | 68%–100% | **7.9** | 8.0 | **22,685** | 21,589 | 2,379 | 0 |
| **executor_only + shell-only** | 8/8 | 68%–100% | 8.5 | 8.0 | 22,694 | 16,962 | 2,320 | 0 |
| **native_todo** | 8/8 | 68%–100% | 10.0 | 10.5 | 157,446 | 57,043 | 2,732 | 0 |
| **native_todo + shell-only** | 8/8 | 68%–100% | 10.1 | 9.0 | 124,223 | 41,250 | 2,969 | 0 |

全部 32 trial：`reward=1`，`stop_reason=completed`，artifact `ok`，0 异常。

## 结论

1. **通过率没有区分度。** 四臂都是 8/8，Wilson 区间完全相同。按 `PROJECT.md` 的既有判断，这 4 题对 `deepseek-v4-flash` 已饱和，`n=2` 只能发现大效应。**不能**据此说 `native_todo` 更好或更差。
2. **代价有稳定差异。** 原生 todo 让平均轮次 7.9 → 10.0（+27%）、平均输入 token 22.7k → 157.4k（+594%）。缓存命中率各臂都在 84%–87%，所以不是缓存失效导致的。
3. **均值被离群试次放大。** `native_todo` 的单次输入 token 最高 500,591（`h2`，14 轮），`native_todo + shell-only` 最高 470,612（`h2`，12 轮）；中位数 57k / 41k 更能代表常态，但仍显著高于基准的 21.6k。
4. **shell-only 不是瓶颈。** 在两种拓扑下它都不降通过率；`executor_only` 里它甚至略降输入中位数（21.6k → 17.0k），只有 `h4` 出现一次 17 轮 / 67.7k 的离群。

即：**当前默认（原生 todo）用 6 倍输入 token 换来了相同的通过率**，唯一可测的收益是让计划对外可见（`write_todos` 的 todo 会流进 trace 与 CLI 面板）。若目标是做题效率，`executor_only` 仍是更优默认；若目标是可观测的计划，应先解决它为什么在 `h2` / `h4` 上反复重写计划（大量上下文重放），再考虑设为默认。

## 逐题明细

### executor_only — `jobs/final-executor-only`

| task | level | reward | turns | in tok | out tok | stop_reason |
|---|---|---|---|---|---|---|
| h1-order-desk | H1 | 1 | 6 | 14,579 | 1,333 | completed |
| h1-order-desk | H1 | 1 | 9 | 29,914 | 2,837 | completed |
| h2-mail-room | H2 | 1 | 7 | 18,951 | 2,266 | completed |
| h2-mail-room | H2 | 1 | 8 | 20,543 | 2,577 | completed |
| h3-stock-sync | H3 | 1 | 9 | 30,400 | 3,084 | completed |
| h3-stock-sync | H3 | 1 | 8 | 22,635 | 2,416 | completed |
| h4-plan-billing | H4 | 1 | 7 | 16,975 | 1,895 | completed |
| h4-plan-billing | H4 | 1 | 9 | 27,485 | 2,623 | completed |

### executor_only + shell-only — `jobs/final-executor-only-shell`

| task | level | reward | turns | in tok | out tok | stop_reason |
|---|---|---|---|---|---|---|
| h1-order-desk | H1 | 1 | 8 | 19,784 | 2,143 | completed |
| h1-order-desk | H1 | 1 | 7 | 16,139 | 2,056 | completed |
| h2-mail-room | H2 | 1 | 6 | 11,168 | 2,346 | completed |
| h2-mail-room | H2 | 1 | 8 | 16,999 | 2,887 | completed |
| h3-stock-sync | H3 | 1 | 8 | 22,098 | 2,305 | completed |
| h3-stock-sync | H3 | 1 | 8 | 16,926 | 1,835 | completed |
| h4-plan-billing | H4 | 1 | 17 | 67,666 | 3,179 | completed |
| h4-plan-billing | H4 | 1 | 6 | 10,769 | 1,808 | completed |

### native_todo — `jobs/final-native-todo`

| task | level | reward | turns | in tok | out tok | stop_reason |
|---|---|---|---|---|---|---|
| h1-order-desk | H1 | 1 | 12 | 57,554 | 2,672 | completed |
| h1-order-desk | H1 | 1 | 11 | 50,717 | 2,295 | completed |
| h2-mail-room | H2 | 1 | 9 | 271,898 | 2,449 | completed |
| h2-mail-room | H2 | 1 | 14 | 500,591 | 3,392 | completed |
| h3-stock-sync | H3 | 1 | 12 | 56,532 | 3,362 | completed |
| h3-stock-sync | H3 | 1 | 6 | 19,882 | 1,768 | completed |
| h4-plan-billing | H4 | 1 | 6 | 21,397 | 2,281 | completed |
| h4-plan-billing | H4 | 1 | 10 | 280,997 | 3,640 | completed |

### native_todo + shell-only — `jobs/final-native-todo-shell`

| task | level | reward | turns | in tok | out tok | stop_reason |
|---|---|---|---|---|---|---|
| h1-order-desk | H1 | 1 | 9 | 268,628 | 1,872 | completed |
| h1-order-desk | H1 | 1 | 8 | 27,689 | 2,085 | completed |
| h2-mail-room | H2 | 1 | 6 | 17,670 | 1,925 | completed |
| h2-mail-room | H2 | 1 | 12 | 470,612 | 3,829 | completed |
| h3-stock-sync | H3 | 1 | 8 | 25,788 | 2,134 | completed |
| h3-stock-sync | H3 | 1 | 9 | 28,004 | 2,106 | completed |
| h4-plan-billing | H4 | 1 | 12 | 54,497 | 4,399 | completed |
| h4-plan-billing | H4 | 1 | 17 | 100,894 | 5,402 | completed |

## 第二轮：加入 todo 工作流提示词后（v2）

改动只有提示词（`design.md` 的「提示词：todo 工作流（软约束）」）：装了 `TodoListMiddleware` 的臂多一段「多步任务先写 3–6 条 todo、随进展改状态」；`shell_anything` 的 “only tool” 改成 “only tool **for doing work**”，不再顺带排除 `write_todos`。同机、同日、同题集、同 `-k 2 -n 4` 重跑四臂，命名 `p2-*`。

| 臂 | 通过 | 平均轮次 | 平均输入 tok | 输入中位 | 平均输出 tok |
|---|---|---|---|---|---|
| **executor_only** | 8/8 | 8.8 | 56,570 | 28,508 | 2,530 |
| **executor_only + shell-only** | 8/8 | 7.6 | 18,578 | 18,168 | 2,365 |
| **native_todo** | 8/8 | 11.5 | 54,295 | 49,830 | 2,972 |
| **native_todo + shell-only** | 8/8 | 8.5 | 30,623 | 27,316 | 2,469 |

32 trial 全部 `reward=1`、`stop_reason=completed`、artifact `ok`、0 异常。

### 与第一轮对比

| 臂 | 通过 | 轮次 | 输入均值 | 输入中位 |
|---|---|---|---|---|
| executor_only | 8/8 → 8/8 | 7.9 → 8.8 | 22,685 → 56,570 | 21,589 → 28,508 |
| executor_only + shell-only | 8/8 → 8/8 | 8.5 → 7.6 | 22,694 → 18,578 | 16,962 → 18,168 |
| native_todo | 8/8 → 8/8 | 10.0 → 11.5 | 157,446 → **54,295** | 57,043 → 49,830 |
| native_todo + shell-only | 8/8 → 8/8 | 10.1 → 8.5 | 124,223 → **30,623** | 41,250 → 27,316 |

三点：

1. **离群尾部被削掉。** 第一轮 `native_todo` 有 271k / 500k / 281k 三次试次（反复重写计划导致上下文重放），第二轮 8 次试次最高 88.6k，均值因此从 157.4k 降到 54.3k；中位只降 13%（57.0k → 49.8k），说明改善主要来自「不再爆掉」，不是「普遍变省」。
2. **同批内的相对代价基本拉平。** 第一轮 native_todo 输入是 executor_only 的 6.9 倍，第二轮均值 0.96 倍、中位 1.75 倍。但这一条**不能全记在提示词头上**：`executor_only` 的提示词没改，均值却从 22.7k 涨到 56.6k —— 它这 8 次里有一次 `h2` 259,923（剔除后均值 27.5k）。同日单跑一臂的方差本身就有 ±2 倍量级，所以「6.9 倍 → 0.96 倍」里有多少是提示词、多少是噪声，n=8 分不开。
3. **轮次没有降。** native_todo 10.0 → 11.5，多写计划多花约 1.5 轮；`native_todo + shell-only` 反而 10.1 → 8.5。方向不一致，视为噪声。

### 未验证

- **`write_todos` 触发率没有可靠数字。** 16 份 native 试次的 `agent/langgraph-run.log` 里有 11 份是 0 字节（Harbor 的 `docker compose cp` 对 `jobs/<job>/...` 相对路径解析失败，见 `trial.log` 的 `invalid output path`），只剩 5 份可读，而 **这 5 份全都打印了 todo 面板**。样本不足以证明触发率上升，只能说没有反例。
- 通过率仍无区分度（四臂 8/8，Wilson 区间相同），4 题对 `deepseek-v4-flash` 依旧饱和。

## 复现

```bash
# 汇总一个 job 目录 → eval/<arm>.jsonl
python3 projects/terminal-coding-agent/scripts/collect_eval_results.py jobs/<job> \
  --out projects/terminal-coding-agent/eval/<arm>.jsonl
```

原始 trial 目录：第一轮 `jobs/final-executor-only`、`jobs/final-executor-only-shell`、`jobs/final-native-todo`、`jobs/final-native-todo-shell`；第二轮 `jobs/p2-executor-only`、`jobs/p2-executor-only-shell`、`jobs/p2-native-todo`、`jobs/p2-native-todo-shell`。

> 注：`cost_rmb` 全为 0 —— `config.py` 里的价格字段已按用户要求清空，本表不报成本。
