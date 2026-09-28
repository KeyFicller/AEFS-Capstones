# hello-agent

Hello-world 级模板：**示意符合 `.cursor/rules/` 的完成态项目长什么样**。内容不重要，结构才重要。

环境是**仓库级共享**的：一个 `.venv` 放在仓库根，所有项目通用，依赖统一写在根 `requirements.txt`。

## 跑起来（在仓库根目录）

```bash
uv venv                              # 建共享虚拟环境（只需一次）
uv pip install -r requirements.txt  # 安装全部依赖

uv run --no-project pytest examples/hello-agent/tests -q   # 测试
uv run --no-project hello-agent run "say hello"            # 正常完成
uv run --no-project hello-agent run "x" --loop-forever     # 演示 turn 熔断
uv run --no-project python examples/hello-agent/eval/run_eval.py  # 评测
```

## 结构（对照 rules）

| 路径 | 对应规则 |
|---|---|
| `requirements.txt`（仓库根） | `60-python-conventions`：全局依赖唯一来源 |
| `PROJECT.md` | `10-project-profile`：项目特有信息 |
| `src/hello_agent/loop.py` | `30-agent-architecture`：plan/act/observe/recover |
| `src/hello_agent/tools.py` | `20-python-stack`：工具是唯一副作用边界 |
| `tests/` | `60-python-conventions`、`70-eval-and-verification` |
| `eval/` | `70-eval-and-verification`：指标落盘 |

## 这是模板，不是成品

真实项目在此基础上替换：模型层（LangGraph + OpenRouter）、沙箱（`40`）、
可观测性（`50`）、真实评测集与基线（`70`）。
