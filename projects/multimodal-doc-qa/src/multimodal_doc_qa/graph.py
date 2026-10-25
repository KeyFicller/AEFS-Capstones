"""LangGraph assembly of the bounded agentic RAG loop.

``rounds`` is incremented in ``retrieve`` only. Counting it in two nodes lets a run pass
``max_rounds``. An empty page pool goes to ``recover`` and is never shown to the answerer.
``stop_reason`` stays empty on a normal finish.
"""

import asyncio
import functools
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from augment import build_augment_graph, parse_techniques
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState

from multimodal_doc_qa import aio
from multimodal_doc_qa.agent import nodes
from multimodal_doc_qa.config import Settings
from multimodal_doc_qa.limits import Budget
from multimodal_doc_qa.retrievers.contract import documents_to_hits
from multimodal_doc_qa.schemas import Answer, Document
from multimodal_doc_qa.synth.answer import prompt_messages

logger = logging.getLogger(__name__)

# One rerank call per query, capped so a burst of queries cannot open dozens of TLS
# connections at the provider at once. The uplink saturates at two 12.5 MB requests in
# flight; four is the measured wall-clock minimum.
RERANK_CONCURRENCY = 4


class AskState(MessagesState):
    """One ask. ``messages`` is the conversation; the other fields are this loop only."""

    queries: list[str]
    page_ids: list[str]
    rounds: int
    answer: dict | None
    unsupported: list[str]
    stop_reason: str
    intent: str


def initial_state(question: str, history: Sequence[BaseMessage] = ()) -> AskState:
    """The state every ask starts from. One question, one loop.

    ``history`` is earlier turns as messages. This question is appended as a
    ``HumanMessage``. Callers must not hand-build the state.
    """
    return {
        "messages": [*history, HumanMessage(content=question)],
        "queries": [],
        "page_ids": [],
        "rounds": 0,
        "answer": None,
        "unsupported": [],
        "stop_reason": "",
        "intent": "",
    }


@dataclass
class GraphDeps:
    """Everything the nodes need that is not in state.

    Every model must be schema-bound except ``augment_model``: the augment component binds
    ``Queries`` itself, so it is handed a raw chat model.
    """

    retriever: Any
    augment_model: Any
    assessor_model: Any
    verifier_model: Any
    synth: Any
    render_dir: Path
    documents: dict[str, Document] | None = None
    budget: Budget | None = None
    rerank: Any = None
    classifier: Any = None


def rerank_pairs(
    pairs: list[tuple[str, list]],
    rerank: Callable[..., Any],
    budget: "Budget",
) -> list[tuple[str, list]]:
    """Rerank each ``(query, docs)`` pair, ``RERANK_CONCURRENCY`` calls in flight at most.

    Returns the pairs in the order they were given. A pair with fewer than two pages has
    nothing to reorder, and a spent budget leaves the remaining pairs as retrieved:
    retrieval still happened, only the reordering is skipped.

    Runs on the shared loop (``aio.run``), so the rerank model's pooled connections stay
    valid across hops and across threads.
    """
    chosen: list[int] = []
    for index, (_, docs) in enumerate(pairs):
        if len(docs) < 2:
            continue
        if budget.exhausted():
            break
        # Charged before dispatch: gather completes out of order, so a mid-flight charge
        # would make the call count depend on which query finished first.
        budget.spend_call()
        chosen.append(index)
    if not chosen:
        return pairs

    async def run_pairs(indices: list[int]) -> list[list]:
        semaphore = asyncio.Semaphore(RERANK_CONCURRENCY)

        async def one(index: int) -> list:
            async with semaphore:
                query, docs = pairs[index]
                return await rerank(query, docs)

        return list(await asyncio.gather(*(one(index) for index in indices)))

    ranked = aio.run(run_pairs(chosen))
    out = list(pairs)
    for index, docs in zip(chosen, ranked, strict=True):
        out[index] = (pairs[index][0], docs)
    return out


def build_graph(deps: GraphDeps, settings: Settings):
    """Compile the ask loop.

    A spent budget ends the run in place and keeps an answer already produced.
    ``rounds`` never exceeds ``settings.max_rounds``.
    """
    budget = deps.budget if deps.budget is not None else Budget.from_settings(settings)

    def metered(node: Callable[[AskState], dict]) -> Callable[[AskState], dict]:
        """Charge one call before the node runs; a spent budget ends the run instead.

        The guard lives here so no node can forget it. It runs *before* the node body,
        so an exhausted budget never half-runs a node.
        """

        @functools.wraps(node)
        def guarded(state: AskState) -> dict:
            if budget.exhausted():
                return {"stop_reason": "budget_exhausted"}
            budget.spend_call()
            return node(state)

        return guarded

    def retrieve_node(state: AskState) -> dict:
        # The entry hop bypasses the assess/verify guards, so the bound has to hold here too.
        if state["rounds"] >= settings.max_rounds:
            return {}
        pool = list(dict.fromkeys(state["page_ids"]))
        # Retrieval is serial: the retriever is CPU-bound and the encoder holds a lock.
        pairs = [(query, list(deps.retriever.invoke(query))) for query in state["queries"]]
        if deps.rerank is not None:
            pairs = rerank_pairs(pairs, deps.rerank, budget)
        for _, docs in pairs:
            # An injected retriever may omit page_id; that is a non-answer, not a crash.
            hits = documents_to_hits([d for d in docs if d.metadata.get("page_id") is not None])
            for hit in hits:
                if hit.page_id not in pool:
                    pool.append(hit.page_id)
        return {"page_ids": pool, "rounds": state["rounds"] + 1}

    @metered
    def assess_node(state: AskState) -> dict:
        followups = nodes.assess(
            deps.assessor_model,
            state["messages"],
            state["page_ids"],
            deps.render_dir,
            deps.documents,
        )
        return {"queries": followups}

    @metered
    def synthesize_node(state: AskState) -> dict:
        answer = deps.synth.synthesize(
            state["messages"], state["page_ids"], deps.render_dir, deps.documents
        )
        # A structured reply can parse to nothing; that is an empty answer, not a crash.
        # ``verify`` cannot check it, so the ask ends here with the reason recorded.
        if answer is None:
            return {"answer": None, "unsupported": [], "stop_reason": "synth_empty"}
        # A fresh answer invalidates any earlier verification; leaving the old list would
        # pair this answer with verdicts that belong to a superseded one.
        return {"answer": answer.model_dump(), "unsupported": []}

    @metered
    def verify_node(state: AskState) -> dict:
        answer = Answer.model_validate(state["answer"])
        unsupported = nodes.verify(
            deps.verifier_model,
            state["messages"],
            answer,
            state["page_ids"],
            deps.render_dir,
            deps.documents,
        )
        return {"unsupported": unsupported, "queries": unsupported}

    def recover_node(state: AskState) -> dict:
        if not state["page_ids"]:
            return {"stop_reason": "recover_empty"}
        return {"stop_reason": "recover_exhausted"}

    @metered
    def intent_node(state: AskState) -> dict:
        label = "work"
        try:
            label = deps.classifier.classify(state["messages"]).intent
        except Exception:  # noqa: BLE001 - an unclear turn is work, never a crash
            logger.exception("intent classification failed; defaulting to work")
        return {"intent": label}

    @metered
    def chat_node(state: AskState) -> dict:
        answer = deps.synth.reply(state["messages"])
        return {"answer": answer.model_dump()}

    def after_intent(state: AskState) -> str:
        return "chat" if state["intent"] == "chat" else "augment"

    def after_retrieve(state: AskState) -> str:
        return "recover" if not state["page_ids"] else "assess"

    def after_augment(state: AskState) -> str:
        return END if state["stop_reason"] else "retrieve"

    def after_assess(state: AskState) -> str:
        if state["stop_reason"]:
            return END
        if state["queries"] and state["rounds"] < settings.max_rounds:
            return "retrieve"
        return "synthesize"

    def after_synthesize(state: AskState) -> str:
        return END if state["stop_reason"] else "verify"

    def after_verify(state: AskState) -> str:
        if state["stop_reason"]:
            return END
        if state["unsupported"] and state["rounds"] < settings.max_rounds:
            return "retrieve"
        if state["unsupported"]:
            return "recover"
        return END

    graph = StateGraph(AskState)
    graph.add_node(
        "augment",
        build_augment_graph(
            deps.augment_model,
            techniques=parse_techniques(settings.augment),
            budget=budget,
            render=prompt_messages,
        ),
    )
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("assess", assess_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("verify", verify_node)
    graph.add_node("recover", recover_node)
    if deps.classifier is not None:
        graph.add_node("intent", intent_node)
        graph.add_node("chat", chat_node)
        graph.add_edge("chat", END)
        graph.add_edge(START, "intent")
        graph.add_conditional_edges("intent", after_intent, {"chat": "chat", "augment": "augment"})
    else:
        graph.add_edge(START, "augment")
    graph.add_conditional_edges("augment", after_augment, {"retrieve": "retrieve", END: END})
    graph.add_conditional_edges(
        "retrieve", after_retrieve, {"recover": "recover", "assess": "assess"}
    )
    graph.add_conditional_edges(
        "assess", after_assess, {"retrieve": "retrieve", "synthesize": "synthesize", END: END}
    )
    graph.add_conditional_edges("synthesize", after_synthesize, {"verify": "verify", END: END})
    graph.add_conditional_edges(
        "verify", after_verify, {"retrieve": "retrieve", "recover": "recover", END: END}
    )
    graph.add_edge("recover", END)

    compiled = graph.compile()
    return compiled.with_config({"callbacks": [budget.callback()], "run_name": "mdq-ask"})


class _DiagramModel:
    """A stand-in for the augment model. The component binds it at build time; nothing calls it."""

    def with_structured_output(self, schema: object) -> "_DiagramModel":
        return self


if __name__ == "__main__":
    settings = Settings()
    png_path = Path(__file__).resolve().parents[2] / "graph.png"
    structure = GraphDeps(
        retriever=None,
        augment_model=_DiagramModel(),
        assessor_model=None,
        verifier_model=None,
        synth=None,
        render_dir=png_path.parent,
        # Non-None so the diagram shows the intent gate; no call is ever made.
        classifier=object(),
    )
    png_path.write_bytes(build_graph(structure, settings).get_graph(xray=True).draw_mermaid_png())
    print(f"graph     {png_path}")  # noqa: T201  # noqa: T201
