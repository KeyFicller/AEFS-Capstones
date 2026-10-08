"""Front gate: classify the turn as chat or work before the pipeline runs."""

import logging
from collections.abc import Callable
from typing import Any

from telemetry import (
    chat_span,
    record_chat_usage,
    resolve_model_name,
)

from terminal_coding_agent.ledger import BudgetSession
from terminal_coding_agent.models import AgentModels
from terminal_coding_agent.state import CodingAgentState

logger = logging.getLogger(__name__)

WORK_HINT = (
    "the agent must plan and act inside the worktree "
    "(read / write files, run commands, change code, debug)"
)


def build_intent(models: AgentModels) -> Callable[[CodingAgentState], dict[str, Any]]:
    """Return the `intent` node.

    The component is imported inside the node, so a run that does not enable the gate
    (Harbor) never imports `intent` and never needs it installed in the container.
    """
    model_name = resolve_model_name(models.planner)

    def intent(state: CodingAgentState) -> dict[str, Any]:
        from intent import Classifier

        label = "work"
        with BudgetSession(state) as budget:
            try:
                with chat_span(model_name) as span:
                    verdict = Classifier(models.planner, work_hint=WORK_HINT).classify(
                        list(state["messages"])
                    )
                    record_chat_usage(span, verdict.raw, model=model_name)
                budget.observe(verdict.raw)
                label = verdict.intent
            except Exception:  # noqa: BLE001 - an unclear turn is work, never a crash
                logger.exception("intent classification failed; defaulting to work")
        return {"intent": label, **budget.updates()}

    return intent
