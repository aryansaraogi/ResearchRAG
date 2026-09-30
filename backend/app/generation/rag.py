"""End-to-end RAG: retrieve -> rerank -> citation-grounded generation."""

from __future__ import annotations

from collections.abc import Iterator

from pydantic import BaseModel

from app.generation import citations as cit
from app.generation.llm import get_llm
from app.generation.prompts import ANSWER_SYSTEM, NOT_FOUND, answer_prompt
from app.retrieval.filters import SearchFilters
from app.retrieval.hybrid import RetrievalMode, RetrievedChunk, search


class RAGAnswer(BaseModel):
    answer: str
    citations: list[cit.Citation]
    sources: list[RetrievedChunk]


def retrieve(question: str, filters: SearchFilters | None, mode: RetrievalMode, rerank: bool,
             top_k: int | None) -> list[RetrievedChunk]:
    return search(question, filters=filters, mode=mode, rerank=rerank, top_k=top_k)


def _finalize(raw: str, sources: list[RetrievedChunk]) -> RAGAnswer:
    text = cit.strip_invalid(raw.strip(), len(sources))
    return RAGAnswer(answer=text, citations=cit.resolve(text, sources), sources=sources)


def answer(
    question: str,
    filters: SearchFilters | None = None,
    history: list[dict] | None = None,
    mode: RetrievalMode = RetrievalMode.hybrid,
    rerank: bool = True,
    top_k: int | None = None,
) -> RAGAnswer:
    sources = retrieve(question, filters, mode, rerank, top_k)
    if not sources:
        return RAGAnswer(answer=NOT_FOUND, citations=[], sources=[])
    raw = get_llm().generate(answer_prompt(question, sources, history), system=ANSWER_SYSTEM)
    return _finalize(raw, sources)


def answer_stream(
    question: str,
    filters: SearchFilters | None = None,
    history: list[dict] | None = None,
    mode: RetrievalMode = RetrievalMode.hybrid,
    rerank: bool = True,
    top_k: int | None = None,
) -> Iterator[tuple[str, object]]:
    """Yields ("sources", [...]), then ("token", str)*, then ("done", RAGAnswer)."""
    sources = retrieve(question, filters, mode, rerank, top_k)
    yield "sources", [s.model_dump() for s in sources]
    if not sources:
        yield "token", NOT_FOUND
        yield "done", RAGAnswer(answer=NOT_FOUND, citations=[], sources=[])
        return
    parts: list[str] = []
    for token in get_llm().stream(answer_prompt(question, sources, history), system=ANSWER_SYSTEM):
        parts.append(token)
        yield "token", token
    yield "done", _finalize("".join(parts), sources)
