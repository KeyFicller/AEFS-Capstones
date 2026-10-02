"""plan -> act -> observe -> recover 的最小落地，外加硬性 turn 上限。"""

from dataclasses import dataclass, field

from hello_agent.tools import TOOLS

MAX_TURNS = 3


@dataclass
class RunResult:
    task: str
    plan: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    turns: int = 0
    tokens: int = 0
    cost: float = 0.0
    done: bool = False


def _model_step(step: int, loop_forever: bool) -> tuple[str, str | None]:
    """假模型：第 0 步给计划，之后调用工具；`loop_forever` 时永不收敛。"""
    if step == 0:
        return "plan", None
    if loop_forever:
        return "act", "world"
    if step == 1:
        return "act", "world"
    return "done", None


def run(task: str, *, loop_forever: bool = False) -> RunResult:
    result = RunResult(task=task)
    for step in range(MAX_TURNS):
        result.turns += 1
        result.tokens += 10
        result.cost += 0.001

        thought, argument = _model_step(step, loop_forever)
        if thought == "plan":
            result.plan = [task]  # 整体重写，不做增量 mutate
        elif thought == "act":
            result.observations.append(TOOLS["say_hello"](argument or "").output)
        else:
            result.done = True
            break

    # 循环自然结束但未 done => 触达 turn 上限，按 recover 处理而非崩溃
    return result
