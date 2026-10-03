"""End-to-end RAG: rewrite follow-ups -> retrieve -> rerank -> citation-grounded generation."""

from __future__ import annotations

import logging
from collections.abc import Iterator

from pydantic import BaseModel

from app.generation import citations as cit
from app.generation.llm import LLMNotConfigured, get_llm
from app.generation.prompts import ANSWER_SYSTEM, CONDENSE_SYSTEM, NOT_FOUND, answer_prompt, condense_prompt
from app.retrieval.filters import SearchFilters
from app.retrieval.hybrid import RetrievalMode, RetrievedChunk, search

log = logging.getLogger(__name__)


class RAGAnswer(BaseModel):
    answer: str
    citations: list[cit.Citation]
    sources: list[RetrievedChunk]
    query: str | None = None  # what retrieval searched for, when a follow-up was rewritten
    model: str | None = None  # the Gemini model that wrote the answer


def standalone_question(question: str, history: list[dict] | None) -> str:
    """Retrieval sees only one string, so a follow-up like "what are its limitations?" would search for
    "its limitations". Rewrite it with the conversation; on any failure, search the question as asked."""
    if not history:
        return question
    try:
        raw = get_llm().generate(condense_prompt(question, history), system=CONDENSE_SYSTEM,
                                 temperature=0.0, quick=True)
    except LLMNotConfigured:
        raise
    except Exception as e:  # noqa: BLE001 - a failed rewrite shouldn't block the answer
        log.warning("Follow-up rewrite failed (%s); searching the question as asked", e)
        return question
    lines = [ln.strip().strip('"“”').strip() for ln in raw.strip().splitlines() if ln.strip()]
    rewritten = lines[0] if lines else ""
    return rewritten if 3 <= len(rewritten) <= 400 else question


def retrieve(question: str, filters: SearchFilters | None, mode: RetrievalMode, rerank: bool,
             top_k: int | None) -> list[RetrievedChunk]:
    return search(question, filters=filters, mode=mode, rerank=rerank, top_k=top_k)


def _finalize(raw: str, sources: list[RetrievedChunk], query: str | None, model: str | None) -> RAGAnswer:
    text = cit.strip_invalid(raw.strip(), len(sources))
    return RAGAnswer(answer=text, citations=cit.resolve(text, sources), sources=sources, query=query, model=model)


def answer(
    question: str,
    filters: SearchFilters | None = None,
    history: list[dict] | None = None,
    mode: RetrievalMode = RetrievalMode.hybrid,
    rerank: bool = False,
    top_k: int | None = None,
) -> RAGAnswer:
    query = standalone_question(question, history)
    rewritten = query if query != question else None
    sources = retrieve(query, filters, mode, rerank, top_k)
    if not sources:
        return RAGAnswer(answer=NOT_FOUND, citations=[], sources=[], query=rewritten)
    info: dict = {}
    raw = get_llm().generate(answer_prompt(question, sources, history), system=ANSWER_SYSTEM, info=info)
    return _finalize(raw, sources, rewritten, info.get("model"))


def answer_stream(
    question: str,
    filters: SearchFilters | None = None,
    history: list[dict] | None = None,
    mode: RetrievalMode = RetrievalMode.hybrid,
    rerank: bool = False,
    top_k: int | None = None,
) -> Iterator[tuple[str, object]]:
    """Yields ("query", str) if a follow-up was rewritten, then ("sources", [...]), ("token", str)*,
    and finally ("done", RAGAnswer)."""
    query = standalone_question(question, history)
    rewritten = query if query != question else None
    if rewritten:
        yield "query", rewritten
    sources = retrieve(query, filters, mode, rerank, top_k)
    yield "sources", [s.model_dump() for s in sources]
    if not sources:
        yield "token", NOT_FOUND
        yield "done", RAGAnswer(answer=NOT_FOUND, citations=[], sources=[], query=rewritten)
        return
    info: dict = {}
    parts: list[str] = []
    for token in get_llm().stream(answer_prompt(question, sources, history), system=ANSWER_SYSTEM, info=info):
        parts.append(token)
        yield "token", token
    yield "done", _finalize("".join(parts), sources, rewritten, info.get("model"))
