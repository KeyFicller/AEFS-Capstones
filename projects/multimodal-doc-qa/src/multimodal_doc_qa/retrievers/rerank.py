"""Optional VLM rerank of an already retrieved page list. Off unless the caller asks."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from multimodal_doc_qa.schemas import Document as CatalogDocument
from multimodal_doc_qa.synth.answer import build_page_blocks


class PageRank(BaseModel):
    """Most relevant page id first. Every id the model was shown should appear once."""

    page_ids: list[str] = Field(default_factory=list)


def order_documents(documents: list[Document], page_ids: list[str]) -> list[Document]:
    """Put ``documents`` in ``page_ids`` order. A page the model omitted keeps its relative place at the end."""
    rank = {page_id: index for index, page_id in enumerate(page_ids)}
    tail = len(rank)
    return sorted(documents, key=lambda doc: rank.get(doc.metadata.get("page_id"), tail))


def rerank_pages(
    model: object,
    query: str,
    documents: list[Document],
    render_dir: Path,
    catalog: dict[str, CatalogDocument] | None,
) -> list[Document]:
    """Ask ``model`` to order ``documents`` for ``query``. Fewer than two pages skips the call."""
    page_ids = [doc.metadata["page_id"] for doc in documents if doc.metadata.get("page_id")]
    if len(page_ids) < 2:
        return documents
    content = [
        {"type": "text", "text": f"Order these pages by relevance.\nquestion: {query}"},
        *build_page_blocks(page_ids, render_dir, catalog),
    ]
    ranked = model.invoke([HumanMessage(content=content)])
    return order_documents(documents, list(ranked.page_ids))
