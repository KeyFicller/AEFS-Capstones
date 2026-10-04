"""LangGraph assembly of the bounded agentic RAG loop.

``rounds`` is incremented in ``retrieve`` only. Counting it in two nodes lets a run pass
``max_rounds``. An empty page pool goes to ``recover`` and is never shown to the answerer.
``stop_reason`` stays empty on a normal finish.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState

from multimodal_doc_qa.agent import nodes
from multimodal_doc_qa.budget import Budget
from multimodal_doc_qa.config import Settings
from multimodal_doc_qa.schemas import Answer, Document


class AskState(MessagesState):
    """One ask. ``messages`` is the conversation; the other fields are this loop only."""

    subqueries: list[str]
    page_ids: list[str]
    rounds: int
    answer: dict | None
    unsupported: list[str]
    stop_reason: str


def initial_state(question: str, history: Sequence[BaseMessage] = ()) -> AskState:
    """The state every ask starts from. One question, one loop.

    ``history`` is earlier turns as messages. This question is appended as a
    ``HumanMessage``. Callers must not hand-build the state.
    """
    return {
        "messages": [*history, HumanMessage(content=question)],
        "subqueries": [],
        "page_ids": [],
        "rounds": 0,
        "answer": None,
        "unsupported": [],
        "stop_reason": "",
    }


@dataclass
class GraphDeps:
    """Everything the nodes need that is not in state. Every model must be schema-bound."""

    retriever: Any
    planner_model: Any
    assessor_model: Any
    verifier_model: Any
    synth: Any
    render_dir: Path
    documents: dict[str, Document] | None = None
    budget: Budget | None = None
    rerank: Any = None


def build_graph(deps: GraphDeps, settings: Settings):
    """Compile the ask loop.

    A spent budget ends the run in place and keeps an answer already produced.
    ``rounds`` never exceeds ``settings.max_rounds``.
    """
    budget = deps.budget if deps.budget is not None else Budget.from_settings(settings)

    def plan_node(state: AskState) -> dict:
        if budget.exhausted():
            return {"stop_reason": "budget_exhausted"}
        budget.spend_call()
        return {"subqueries": nodes.plan(deps.planner_model, state["messages"])}

    def retrieve_node(state: AskState) -> dict:
        # The entry hop bypasses the assess/verify guards, so the bound has to hold here too.
        if state["rounds"] >= settings.max_rounds:
            return {}
        pool = list(dict.fromkeys(state["page_ids"]))
        for subquery in state["subqueries"]:
            docs = list(deps.retriever.invoke(subquery))
            if deps.rerank is not None and len(docs) > 1 and not budget.exhausted():
                budget.spend_call()
                docs = deps.rerank(subquery, docs)
            for doc in docs:
                # The retriever is injected; one that omits page_id must not abort the ask.
                page = doc.metadata.get("page_id")
                if page is not None and page not in pool:
                    pool.append(page)
        return {"page_ids": pool, "rounds": state["rounds"] + 1}

    def assess_node(state: AskState) -> dict:
        if budget.exhausted():
            return {"stop_reason": "budget_exhausted"}
        budget.spend_call()
        followups = nodes.assess(
            deps.assessor_model,
            state["messages"],
            state["page_ids"],
            deps.render_dir,
            deps.documents,
        )
        return {"subqueries": followups}

    def synthesize_node(state: AskState) -> dict:
        if budget.exhausted():
            return {"stop_reason": "budget_exhausted"}
        budget.spend_call()
        answer = deps.synth.synthesize(
            state["messages"], state["page_ids"], deps.render_dir, deps.documents
        )
        # A fresh answer invalidates any earlier verification; leaving the old list would
        # pair this answer with verdicts that belong to a superseded one.
        return {"answer": answer.model_dump(), "unsupported": []}

    def verify_node(state: AskState) -> dict:
        if budget.exhausted():
            return {"stop_reason": "budget_exhausted"}
        budget.spend_call()
        answer = Answer.model_validate(state["answer"])
        unsupported = nodes.verify(
            deps.verifier_model,
            state["messages"],
            answer,
            state["page_ids"],
            deps.render_dir,
            deps.documents,
        )
        return {"unsupported": unsupported, "subqueries": unsupported}

    def recover_node(state: AskState) -> dict:
        if not state["page_ids"]:
            return {"stop_reason": "recover_empty"}
        return {"stop_reason": "recover_exhausted"}

    def after_retrieve(state: AskState) -> str:
        return "recover" if not state["page_ids"] else "assess"

    def after_plan(state: AskState) -> str:
        return END if state["stop_reason"] else "retrieve"

    def after_assess(state: AskState) -> str:
        if state["stop_reason"]:
            return END
        if state["subqueries"] and state["rounds"] < settings.max_rounds:
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
    graph.add_node("plan", plan_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("assess", assess_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("verify", verify_node)
    graph.add_node("recover", recover_node)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", after_plan, {"retrieve": "retrieve", END: END})
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


if __name__ == "__main__":
    settings = Settings()
    png_path = Path(__file__).resolve().parents[2] / "graph.png"
    structure = GraphDeps(
        retriever=None,
        planner_model=None,
        assessor_model=None,
        verifier_model=None,
        synth=None,
        render_dir=png_path.parent,
    )
    png_path.write_bytes(build_graph(structure, settings).get_graph(xray=True).draw_mermaid_png())
    print(f"graph     {png_path}")
