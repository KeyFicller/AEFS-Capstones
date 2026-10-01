# terminal-coding-agent

一个跑在终端里的 coding agent。你用自然语言提任务，它在隔离的 worktree 里读代码、改文件、
跑命令，交出一份 patch。评测用一组**手写 Harbor 任务**的通过率，不做外部基线对比。

![REPL 实跑](example.png)

上面这次是让它写一个 Python 入门教学脚本：工具调用逐行打印（`edit_file` 带着色 diff），
底部 `Tasks` 面板原地刷新计划，收尾由 agent 自己写总结。

循环骨架：`make_plan` → `start_task` / `run_agent` / `end_task` → `recover`，所有终止路径都汇到 `summary`。

![图结构](graph.png)

## 跑起来

在仓库根目录（环境是仓库级共享的，一个 `.venv` 所有项目通用）：

```bash
uv venv && uv pip install -r requirements.txt   # 只需一次

uv run --no-project terminal-coding-agent       # 多轮 REPL
uv run --no-project pytest projects/terminal-coding-agent/tests -q
```

REPL 默认只能写**临时 worktree**，退出即删，不会碰你的仓库。要真让它改代码：

```bash
uv run --no-project terminal-coding-agent --worktree /path/to/repo
```

`--session <id>` 固定 thread id 可续跑（需连同 `--worktree` 一起传）。

## 工具

`read_file`、`edit_file`、`ripgrep`、`tree_sitter_symbols`、`run_shell`、`git`。

> 给 LLM 装上 shell 之后是真的无敌。别的工具的边界都得你替它想好，`run_shell` 不用——
> 它会自己 `python3 - <<'EOF'` 起个子进程验证刚写的东西，拿 stdout 和退出码当反馈闭环。
> 上限一夜之间从「你 prompt 写得多好」变成「你敢开多大的口子」。

## 目录

| 路径 | 作用 |
| --- | --- |
| `src/terminal_coding_agent/graph.py` | 图装配（plan / act / observe / recover） |
| `src/terminal_coding_agent/cli.py` | 交互入口：REPL |
| `src/terminal_coding_agent/tools/` | 6 个工具，唯一的副作用边界 |
| `src/terminal_coding_agent/ui.py` | 控制台渲染 |
| `harbor_tasks/` | 评测任务集：`l1-`…`l5-` 五档能力题 + `greeter-fix` 烟测 |
| `scripts/` | `verify_harbor_tasks_local.sh`（本地双向验证）、`collect_eval_results.py`（汇总 job 到 `eval/results.jsonl`） |
| `PROJECT.md` | 项目特有信息（架构 / 预算 / 评测口径）的唯一来源 |
| `tests/` | 单测 |

## 评测任务集

`harbor_tasks/` 下 5 个手写 mini-repo 任务，按**定位难度**逐级递进：

| 档 | 任务 | 给 agent 的信息 |
| --- | --- | --- |
| L1 | `l1-shipping-average` | 可见失败测试**点名**目标函数 |
| L2 | `l2-unit-resolution` | 症状在调用方，真因在被调方；另一处也在用被调方 |
| L3 | `l3-metrics-labels` | 只有报错现象，**无可见测试**；报错位置不是元凶 |
| L4 | `l4-cancelled-orders` | 规格散文，可见测试**全绿**；插入点要自己找 |
| L5 | `l5-settings-cache` | 可见测试全绿 + 仓库 README 写明架构不变量；诱饵是就地打补丁 |

每题分「可见」（`environment/` 里的 repo）与「隐藏」（`tests/` 的 verifier）两层。

本地验证（真 Docker，不需要 LLM / Harbor）：

```bash
bash projects/terminal-coding-agent/scripts/verify_harbor_tasks_local.sh
```

跑批并汇总（cwd = 仓库根）：

```bash
export PATH="$HOME/.local/bin:$PATH"
set -a; source local.env; set +a
harbor run -p projects/terminal-coding-agent/harbor_tasks \
  -a langgraph -m deepseek:deepseek-v4-flash \
  --ae DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY" \
  --allow-agent-host api.deepseek.com -k 1 -n 2 \
  --ak project_path="$(pwd)/projects/terminal-coding-agent"

python3 projects/terminal-coding-agent/scripts/collect_eval_results.py <job-dir>
```

2026-10-01 实测：能力题 **5/5 通过**（8m29s / 0.17 元），逐档 L1–L5 全 pass —— 阶梯可证伪，
但对 `deepseek-v4-flash` **已饱和**，本次运行对能力分无区分度。口径与失败分类见 `PROJECT.md`。
