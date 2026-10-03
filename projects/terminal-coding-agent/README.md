# terminal-coding-agent

终端里的 coding agent：你用自然语言提任务，它在隔离 worktree 内读代码、改文件、跑命令，交出一份 patch。

![REPL 实跑](example.png)

循环骨架：`make_plan` → `start_task` / `run_agent` / `end_task` → `recover` → `summary`。Act 只分派工具，Observe 截断输出回填，Recover 把失败当 Observation 重排计划（上界 3 次），所有终止路径都汇到 `summary` 写 trace。

![图结构](graph.png)

## 跑起来

```bash
./start.sh setup                                                  # 共享 .venv + 依赖，只需一次
./start.sh run terminal-coding-agent                              # 多轮 REPL；默认只写临时 worktree，退出即删
./start.sh run terminal-coding-agent --worktree /path/to/repo     # 真改代码；--session <id> 可续跑
./start.sh test projects/terminal-coding-agent                    # 单测
bash projects/terminal-coding-agent/scripts/verify_harbor_tasks_local.sh   # 任务集双向验证（真 Docker，不用 LLM / Harbor）
```

评测跑批（`harbor run`，需要 `local.env` 里的 `DEEPSEEK_API_KEY`）与实测口径见 `PROJECT.md`。
