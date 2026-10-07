# terminal-coding-agent

终端里的 coding agent：你用自然语言提任务，它在隔离 worktree 内读代码、改文件、跑命令，交出一份 patch；要一段内容（写／解释／翻译，或"把某个文件给我看看"）的请求则直接作答，不产生 patch。

![REPL 实跑](example.png)

循环骨架：`make_plan` → `start_task` / `run_agent` / `end_task` → `recover` → `summary`。Act 只分派工具，Observe 截断输出回填，Recover 把失败当 Observation 重排计划（上界 3 次），所有终止路径都汇到 `summary` 写 trace。CLI 另有一条直答短路径：planner 判为 `answer` 的请求走 `make_plan` → `answer` → `summary`，不产生 todo；下图 `graph.png` 是**未开该分支**的默认拓扑。

![图结构](graph.png)

## 跑起来

```bash
./start.sh setup                                                  # 共享 .venv + 依赖，只需一次
./start.sh run terminal-coding-agent                              # 多轮 REPL；默认只写临时 worktree，退出即删
./start.sh run terminal-coding-agent --worktree /path/to/repo     # 真改代码；--session <id> 可续跑
./start.sh test projects/terminal-coding-agent                    # 单测
bash projects/terminal-coding-agent/scripts/verify_harbor_tasks_local.sh   # 任务集双向验证（真 Docker，不用 LLM / Harbor）
```

评测跑批（`harbor run`，凭据放 `local.env`，模型走 `configurable`）的命令与实测结果见 `PROJECT.md` 的「交付物」。
