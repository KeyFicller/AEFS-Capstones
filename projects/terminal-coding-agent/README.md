# terminal-coding-agent

Hello-world 级模板：**示意符合 `.cursor/rules/` 的完成态项目长什么样**。内容不重要，结构才重要。

环境是**仓库级共享**的：一个 `.venv` 放在仓库根，所有项目通用，依赖统一写在根 `requirements.txt`。

## 跑起来（在仓库根目录）

```bash
uv venv                              # 建共享虚拟环境（只需一次）
uv pip install -r requirements.txt  # 安装全部依赖

uv run --no-project pytest projects/terminal-coding-agent/tests -q   # 测试
uv run --no-project terminal-coding-agent                            # 多轮 REPL
uv run --no-project python projects/terminal-coding-agent/eval/run_eval.py  # 评测
```

## 结构（对照 rules）

| 路径 | 对应规则 |
|---|---|
| `requirements.txt`（仓库根） | `60-python-conventions`：全局依赖唯一来源 |
| `PROJECT.md` | `10-project-profile`：项目特有信息 |
| `src/terminal_coding_agent/graph.py` | `30-agent-architecture`：plan/act/observe/recover 装配 |
| `src/terminal_coding_agent/cli.py` | 交互入口：REPL（`--worktree` / `--session`；默认 worktree 为临时目录，退出即删） |
| `src/terminal_coding_agent/session.py` | 会话语义：每轮重置、轮执行 |
| `src/terminal_coding_agent/ui.py` | 控制台渲染（库代码里唯一的 console 出口） |
| `src/terminal_coding_agent/tools/` | `20-python-stack`：工具是唯一副作用边界 |
| `tests/` | `60-python-conventions`、`70-eval-and-verification` |
| `eval/` | `70-eval-and-verification`：指标落盘 |

## 这是模板，不是成品

真实项目在此基础上替换：模型层（LangGraph + OpenRouter）、沙箱（`40`）、
可观测性（`50`）、真实评测集与基线（`70`）。
