"""Conversational query augmentation: one subgraph, four prompt techniques.

The host supplies a raw chat model, the technique set, a spend ledger, and the way history
is rendered into a prompt. Nothing here knows about a document corpus.
"""

from collections.abc import Callable, Sequence
from typing import Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

TECHNIQUES = ("rewrite", "decompose", "multiquery", "hyde")
MULTIQUERY_N = 3

Render = Callable[[str, Sequence[BaseMessage]], list[BaseMessage]]


class Queries(BaseModel):
    """A technique's output: the retrieval queries it contributes."""

    queries: list[str] = Field(default_factory=list)


class AugmentState(TypedDict):
    """The subgraph's own state. A subset of the host's, so it mounts without translation."""

    messages: list[BaseMessage]
    queries: list[str]
    stop_reason: str


_AUGMENT_SYS = (
    "Rewrite the user's latest question into one standalone retrieval query. Resolve "
    "pronouns and omitted subjects from the conversation so the query stands on its own "
    "without the history. Keep the question's language. Do not answer it."
)
_DECOMPOSE_SYS = (
    "Split the given retrieval query into the minimal set of independent sub-queries "
    "needed to answer it. Each sub-query must stand alone. Return an empty list when the "
    "query has a single information need."
)
_MULTIQUERY_SYS = (
    f"Write {MULTIQUERY_N} different retrieval queries for the given query. Vary the "
    "wording and the terms you use. Each must stand alone. Do not answer the query."
)
_HYDE_SYS = (
    "Write a short passage that would answer the given query if it appeared in a "
    "document. Use the vocabulary such a document would use. Do not answer the user and "
    "do not add commentary."
)


def parse_techniques(spec: str) -> tuple[str, ...]:
    """``spec`` as a canonical-order tuple. An unknown name raises rather than being dropped.

    The order is ``TECHNIQUES``, not the order they were written in: the merge order is a
    property of the design, so a config string must not be able to reorder it.
    """
    named = {part.strip() for part in spec.split(",") if part.strip()}
    unknown = sorted(named - set(TECHNIQUES))
    if unknown:
        raise ValueError(f"unknown augment technique(s): {', '.join(unknown)}")
    return tuple(name for name in TECHNIQUES if name in named)


def latest_human_text(messages: Sequence[BaseMessage]) -> str:
    """Text of the last human message: the bare turn, used when a technique produces nothing."""
    for message in reversed(messages):
        if message.type != "human":
            continue
        content = message.content
        if isinstance(content, str):
            return content
        return "\n".join(
            part["text"]
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def merge_queries(*groups: Sequence[str]) -> list[str]:
    """Concatenate, drop blanks and case-insensitive repeats, keep the first spelling and order."""
    seen: set[str] = set()
    merged: list[str] = []
    for group in groups:
        for raw in group:
            query = raw.strip()
            if query and query.lower() not in seen:
                seen.add(query.lower())
                merged.append(query)
    return merged


def _plain(system: str, messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """The default render: the system prompt and the conversation, untrimmed."""
    return [SystemMessage(content=system), *messages]


def rewrite(model: Any, messages: Sequence[BaseMessage], render: Render = _plain) -> str:
    """One standalone retrieval query for the latest turn; empty when the model produced none."""
    result = model.invoke(render(_AUGMENT_SYS, messages))
    queries = list(result.queries)
    return queries[0] if queries else ""


def decompose(model: Any, query: str) -> list[str]:
    """Independent sub-queries of ``query``; empty when it has a single information need."""
    return _expand(model, _DECOMPOSE_SYS, query)


def multiquery(model: Any, query: str) -> list[str]:
    """Rewordings of ``query``."""
    return _expand(model, _MULTIQUERY_SYS, query)


def hyde(model: Any, query: str) -> list[str]:
    """A passage that would answer ``query``."""
    return _expand(model, _HYDE_SYS, query)


def _expand(model: Any, system: str, query: str) -> list[str]:
    """One structured call over ``query`` alone: an expander does not need the conversation."""
    result = model.invoke([SystemMessage(content=system), HumanMessage(content=query)])
    return list(result.queries)


_EXPANDERS = {"decompose": decompose, "multiquery": multiquery, "hyde": hyde}


def build_augment_graph(
    model: BaseChatModel,
    *,
    techniques: Sequence[str],
    budget: Any,
    render: Render = _plain,
):
    """Bind ``Queries`` once, then wire rewrite -> the enabled expanders -> finalize.

    ``budget`` is anything with ``spend_call()`` and ``exhausted()``. A step that is
    disabled or cannot afford its call is jumped over by the route function, so it is never
    entered -- the diamond in the design is an edge, not a node.
    """
    bound = model.with_structured_output(Queries)

    def next_step(after: str, state: AugmentState) -> str:
        for step in TECHNIQUES[TECHNIQUES.index(after) + 1 :]:
            if step in techniques and not state["stop_reason"] and not budget.exhausted():
                return step
        return "finalize"

    def rewrite_node(state: AugmentState) -> dict:
        # The base is index 0 for the whole subgraph: the expanders read it and append.
        if budget.exhausted():
            return {"stop_reason": "budget_exhausted"}
        base = ""
        if "rewrite" in techniques:
            budget.spend_call()
            base = rewrite(bound, state["messages"], render)
        return {"queries": [base or latest_human_text(state["messages"])]}

    def expander_node(name: str):
        def run(state: AugmentState) -> dict:
            budget.spend_call()
            produced = _EXPANDERS[name](bound, state["queries"][0])
            return {"queries": [*state["queries"], *produced]}

        return run

    def finalize_node(state: AugmentState) -> dict:
        if state["stop_reason"]:
            return {}
        merged = merge_queries(state["queries"])
        return {"queries": merged or [latest_human_text(state["messages"])]}

    subgraph = StateGraph(AugmentState)
    subgraph.add_node("rewrite", rewrite_node)
    for name in _EXPANDERS:
        subgraph.add_node(name, expander_node(name))
    subgraph.add_node("finalize", finalize_node)
    subgraph.add_edge(START, "rewrite")
    order = (*TECHNIQUES, "finalize")
    for name in ("rewrite", "decompose", "multiquery"):
        # Only the steps after this one are reachable, so the diagram shows no edge back.
        targets = {step: step for step in order[order.index(name) + 1 :]}
        subgraph.add_conditional_edges(
            name, lambda state, name=name: next_step(name, state), targets
        )
    subgraph.add_edge("hyde", "finalize")
    subgraph.add_edge("finalize", END)
    return subgraph.compile()
