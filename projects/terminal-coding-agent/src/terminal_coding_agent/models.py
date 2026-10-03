"""Model factory: read provider / model name from config and build chat models."""

from dataclasses import dataclass

from langchain.chat_models import BaseChatModel, init_chat_model
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama

from terminal_coding_agent.config import MODEL

SYSTEM_PROMPTS = {
    "planner": """
    You are a helpful assistant that helps the user to make a plan for the task.
    The user will provide you with a task and you will need to make a plan for the task.
    The plan should be a list of steps that the user can follow to complete the task.
    The plan should be in the following format:
    - Step 1: ...
    - Step 2: ...
    - Step 3: ...
    If the task does not need to be split into steps, just return the task itself.
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
    model_names = {
        role: config.get("configurable", {}).get(role, MODEL) for role in ["planner", "executor"]
    }
    if local_model := config.get("configurable", {}).get("local_model"):
        models = {
            role: ChatOllama(model=local_model, extra_body={"thinking": {"type": "disabled"}})
            for role, _ in model_names.items()
        }

    else:
        models = {
            role: init_chat_model(model_name, extra_body={"thinking": {"type": "disabled"}})
            for role, model_name in model_names.items()
        }
    return AgentModels(planner=models["planner"], executor=models["executor"])
