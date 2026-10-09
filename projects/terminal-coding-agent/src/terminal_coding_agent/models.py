"""Model factory: read provider / model name from config and build chat models."""

from dataclasses import dataclass

from langchain.chat_models import BaseChatModel, init_chat_model
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama

from terminal_coding_agent.config import MODEL

SYSTEM_PROMPTS = {
    "planner": """
    You are the planner for a terminal coding agent. Turn the user's request into a plan.

    `steps` must end with the deliverable itself: never append a separate check or report
    step (fold verification into the producing step). Keep the number of steps minimal.
    """,
    "chat": """
    Answer the user directly in your reply: the content itself (the essay, the
    explanation, the translation, ...) is the deliverable. Do not write it to a file,
    and do not end with a report about the task — end with the content.
    """,
    "executor": """
    You are a helpful assistant that helps the user to execute the task.
    The user will provide you with a task step and you will need to execute.

    If the step cannot be completed after a genuine attempt, call the
    `report_blocked` tool with the failing command and its error instead of
    retrying the same command or guessing.
    """,
    "debug": """
    A live lldb session is available via `debug_start`, `debug_cmd`, `debug_stop`.
    Build the program with `-g -O0` first. `debug_cmd` runs one lldb command
    (`breakpoint set -f main.cpp -l 42`, `run`, `frame variable`, `thread backtrace`,
    `continue`, ...); `run`/`continue` return when the program stops or the timeout
    interrupts them, and the session stays usable either way.
    `debug_start` on a target that is already open keeps the session (breakpoints and
    the stopped program survive), so a later turn can keep inspecting it. Do not call
    `debug_stop` until the debugging is finished.
    """,
}


@dataclass(frozen=True)
class AgentModels:
    """Chat model used by each role."""

    planner: BaseChatModel
    executor: BaseChatModel


def build_models(config: RunnableConfig) -> AgentModels:
    """Build one model per role; the name comes from config["configurable"][role], else MODEL."""
    configurable = config.get("configurable") or {}
    model_names = {role: configurable.get(role) or MODEL for role in ("planner", "executor")}
    if local_model := configurable.get("local_model"):
        models = {
            role: ChatOllama(model=local_model, extra_body={"thinking": {"type": "disabled"}})
            for role in model_names
        }

    else:
        models = {
            role: init_chat_model(model_name, extra_body={"thinking": {"type": "disabled"}})
            for role, model_name in model_names.items()
        }
    return AgentModels(planner=models["planner"], executor=models["executor"])
