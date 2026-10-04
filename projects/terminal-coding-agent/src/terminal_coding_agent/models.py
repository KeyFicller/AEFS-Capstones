"""Model factory: read provider / model name from config and build chat models."""

from dataclasses import dataclass

from langchain.chat_models import BaseChatModel, init_chat_model
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama

from terminal_coding_agent.config import MODEL

SYSTEM_PROMPTS = {
    "planner": """
    You are the planner for a terminal coding agent. Turn the user's request into a plan.

    Decide `mode` first. The deciding question is whether the worktree has to end up
    different:
    - "answer": the user gets the content in the reply and the worktree is NOT changed —
      write / explain / translate / draft / summarize a passage, or show / display /
      print content that already exists (e.g. "show me that file"). Reading files in
      order to answer is fine and still "answer". Do not split the job into steps: return
      the whole job as a single step. The executor puts the content straight in its reply.
    - "work": finishing the job means changing the worktree — create / edit / delete
      files, run commands that mutate state, change code, debug a failure. Then `steps`
      must end with the deliverable itself: never append a separate check or report step
      (fold verification into the producing step). Keep the number of steps minimal.

    If the worktree need not change, choose "answer" — even when the agent must read
    files to produce the reply. When unsure, choose "work": a misclassified answer skips
    changes the task needs.
    """,
    "answer": """
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
