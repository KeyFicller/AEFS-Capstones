# hello-agent

`./start.sh new <name>` 的模板。内容不重要，结构才重要。

环境是**仓库级共享**的：一个 `.venv` 放在仓库根，所有项目通用，依赖统一写在根 `requirements.txt`。

## 跑起来（在仓库根目录）

```bash
./start.sh setup                                     # 建共享虚拟环境 + 装依赖（只需一次）
./start.sh test examples/hello-agent/tests           # 测试
./start.sh run hello-agent world                     # CLI
./start.sh run python examples/hello-agent/eval/run_eval.py  # 评测，写 eval/results.jsonl
```

## 结构

| 路径 | 作用 |
|---|---|
| `pyproject.toml` | 只声明打包与 console script；依赖在根 `requirements.txt` |
| `PROJECT.md` | 项目特有信息唯一来源（`workflow`） |
| `src/hello_agent/core.py` | 库代码，纯函数 |
| `src/hello_agent/cli.py` | console script 入口 |
| `tests/` | 完成前 `pytest` 全绿 |
| `eval/` | 指标落盘、基线对比 |
