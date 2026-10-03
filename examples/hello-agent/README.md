# hello-agent

`./start.sh new <name>` 的模板：内容不重要，结构才重要，架构与文件职责见 `PROJECT.md`。

## 跑起来

```bash
./start.sh setup                                     # 共享 .venv + 依赖，只需一次
./start.sh test examples/hello-agent/tests           # 单测
./start.sh run hello-agent world                     # CLI
./start.sh run python examples/hello-agent/eval/run_eval.py   # 评测，写 eval/results.jsonl
```
