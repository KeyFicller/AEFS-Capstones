"""LangGraph assembly of the bounded agentic RAG loop.

``rounds`` is incremented in ``retrieve`` only. Counting it in two nodes lets a run pass
``max_rounds``. An empty page pool goes to ``recover`` and is never shown to the answerer.
``stop_reason`` stays empty on a normal finish.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import StateGraph, START, END

from multimodal_doc_qa.budget import Budget
from multimodal_doc_qa.config import Settings
from multimodal_doc_qa.agent import nodes
from multimodal_doc_qa.schemas import Answer


class AskState(TypedDict):
    """Loop state. Holds only values that survive a round trip: ids, text, scalars."""

    question: str
    subqueries: list[str]
    page_ids: list[str]
    rounds: int
    answer: dict | None
    unsupported: list[str]
    stop_reason: str


def initial_state(question: str) -> AskState:
    """The state every ask starts from. Callers must not hand-build it."""
    return {
        "question": question,
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
    budget: Budget | None = None


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
        return {"subqueries": nodes.plan(deps.planner_model, state["question"])}

    def retrieve_node(state: AskState) -> dict:
        pool = list(dict.fromkeys(state["page_ids"]))
        for subquery in state["subqueries"]:
            for doc in deps.retriever.invoke(subquery):
                page = doc.metadata["page_id"]
                if page not in pool:
                    pool.append(page)
        return {"page_ids": pool, "rounds": state["rounds"] + 1}

    def assess_node(state: AskState) -> dict:
        if budget.exhausted():
            return {"stop_reason": "budget_exhausted"}
        budget.spend_call()
        followups = nodes.assess(
            deps.assessor_model, state["question"], state["page_ids"], deps.render_dir
        )
        return {"subqueries": followups}

    def synthesize_node(state: AskState) -> dict:
        if budget.exhausted():
            return {"stop_reason": "budget_exhausted"}
        budget.spend_call()
        answer = deps.synth.synthesize(state["question"], state["page_ids"], deps.render_dir)
        return {"answer": answer.model_dump()}

    def verify_node(state: AskState) -> dict:
        if budget.exhausted():
            return {"stop_reason": "budget_exhausted"}
        budget.spend_call()
        answer = Answer.model_validate(state["answer"])
        unsupported = nodes.verify(
            deps.verifier_model, state["question"], answer, state["page_ids"], deps.render_dir
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
    graph.add_conditional_edges("retrieve", after_retrieve, {"recover": "recover", "assess": "assess"})
    graph.add_conditional_edges("assess", after_assess, {"retrieve": "retrieve", "synthesize": "synthesize", END: END})
    graph.add_conditional_edges("synthesize", after_synthesize, {"verify": "verify", END: END})
    graph.add_conditional_edges("verify", after_verify, {"retrieve": "retrieve", "recover": "recover", END: END})
    graph.add_edge("recover", END)

    compiled = graph.compile()
    return compiled.with_config({"callbacks": [budget.callback()], "run_name": "mdq-ask"})


if __name__ == "__main__":
    import os
    import time

    from PIL import Image

    from multimodal_doc_qa.agent.nodes import Followups, Subqueries, Unsupported
    from multimodal_doc_qa.embed.encoder import MultiVectorEncoder
    from multimodal_doc_qa.index.maxsim import MultiVectorIndex
    from multimodal_doc_qa.retrievers.multivector import MultiVectorRetriever
    from multimodal_doc_qa.schemas import page_id
    from multimodal_doc_qa.synth.answer import AnswerSynthesizer, build_chat_model

    _DEMO_QUESTION = "In doc000 page 0, what is stated about EMEA?"
    _DEMO_EXPECTED = "EMEA margin was 16.8%"

    def _write_graph_png(settings: Settings, render_dir: Path, path: Path) -> None:
        """Write the compiled graph's diagram before any weights are loaded."""
        structure = GraphDeps(
            retriever=None,
            planner_model=None,
            assessor_model=None,
            verifier_model=None,
            synth=None,
            render_dir=render_dir,
        )
        compiled = build_graph(structure, settings)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(compiled.get_graph(xray=True).draw_mermaid_png())

    def _build_index(
        model_name: str, settings: Settings, render_dir: Path
    ) -> tuple[MultiVectorEncoder, MultiVectorIndex]:
        """Encode every rendered page. Demo only; production persists the index in ``ingest``."""
        pages = sorted(render_dir.glob("doc*/p*.png"))
        started = time.perf_counter()
        encoder = MultiVectorEncoder(model_name, settings)
        print(
            f"encoder   {type(encoder.model).__name__} "
            f"({model_name}, {settings.device}/{settings.dtype}) "
            f"in {time.perf_counter() - started:.1f}s"
        )

        started = time.perf_counter()
        index = MultiVectorIndex(device="cpu")
        for page in pages:
            vectors = encoder.encode_images([Image.open(page).convert("RGB")])[0]
            index.add(page_id(page.parent.name, int(page.stem[1:])), vectors)
        print(
            f"index     {len(pages)} pages, {index.nbytes() / 1e6:.1f} MB "
            f"in {time.perf_counter() - started:.1f}s"
        )
        return encoder, index

    settings = Settings()
    render_dir = Path(__file__).resolve().parent / "corpus" / "_artifacts"
    png_path = Path(__file__).resolve().parents[2] / "graph.png"

    embedder_model = os.environ.get("MDQ_EMBEDDER_MODEL", settings.embedder_fallback)

    _write_graph_png(settings, render_dir, png_path)
    print(f"graph     {png_path}")
    print()

    encoder, index = _build_index(embedder_model, settings, render_dir)
    total = len(list(render_dir.glob("doc*/p*.png")))

    chat = build_chat_model(settings)
    deps = GraphDeps(
        retriever=MultiVectorRetriever(encoder=encoder, index=index, k=settings.top_k),
        planner_model=chat.with_structured_output(Subqueries),
        assessor_model=chat.with_structured_output(Followups),
        verifier_model=chat.with_structured_output(Unsupported),
        synth=AnswerSynthesizer(settings),
        render_dir=render_dir,
    )

    print()
    started = time.perf_counter()
    out = build_graph(deps, settings).invoke(initial_state(_DEMO_QUESTION))
    elapsed = time.perf_counter() - started

    answer = Answer.model_validate(out["answer"]) if out["answer"] else None
    print(f"question  {_DEMO_QUESTION}")
    print(f"expected  {_DEMO_EXPECTED}")
    print(f"answer    {answer.text if answer else '<none>'}")
    for citation in answer.citations if answer else []:
        print(
            f"cite      {page_id(citation.doc_id, citation.page)}"
            f"  bbox={citation.bbox}"
        )
    print(f"pages     {len(out['page_ids'])}/{total} retrieved  {out['page_ids']}")
    print(f"rounds    {out['rounds']} (max {settings.max_rounds})")
    print(f"unsupported  {out['unsupported']}")
    print(f"stop_reason  {out['stop_reason'] or '<none: finished normally>'}")
    print(f"elapsed   {elapsed:.1f}s")
