---
name: start
description: 仓库根目录的 ./start.sh 是环境、脚手架、测试、运行的统一入口。当需要建虚拟环境 / 装依赖、从模板新建 capstone 项目、跑 pytest、在共享环境里执行命令、清缓存时使用；不要手写等价的 uv / pytest / cp 命令。
---

# ./start.sh

在仓库根执行。所有子命令都基于根 `.venv` + 根 `requirements.txt`（唯一依赖来源）。

| 需求 | 命令 |
|---|---|
| 建 `.venv` 并装依赖 | `./start.sh setup` |
| 从 `examples/hello-agent` 新建项目并注册到 `requirements.txt` | `./start.sh new <name>` |
| 跑测试（默认 `examples/` + `projects/`） | `./start.sh test [path]` |
| 在共享环境里跑任意命令 | `./start.sh run <cmd...>`（等价 `uv run --no-project <cmd>`） |
| 清 `__pycache__` / `.pytest_cache` / `.ruff_cache` | `./start.sh clean` |

## 注意

- `new <name>`：目录 `projects/<name>`，Python 包名 `name` 中 `-` 换 `_`；模板里的 `hello_agent` / `hello-agent` 全部替换。建完要 `./start.sh setup` 重装 editable 包，再 `./start.sh test projects/<name>`。
- `test` 带 `--import-mode=importlib`，多个项目里同名 `test_*.py` 不冲突。单独调 pytest 也要带这个参数。
- console script 找不到时直接调 `.venv/bin/<script>`。
- 需要 `uv`；缺失时脚本自己报错退出。
