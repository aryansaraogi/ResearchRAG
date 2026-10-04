"""End-to-end RAG: plan queries (follow-ups, multi-part questions) -> retrieve -> citation-grounded generation."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator

from pydantic import BaseModel

from app.generation import citations as cit
from app.generation.llm import LLMNotConfigured, get_llm
from app.generation.prompts import ANSWER_SYSTEM, MAX_SUBQUERIES, NOT_FOUND, PLAN_SYSTEM, answer_prompt, plan_prompt
from app.retrieval.filters import SearchFilters
from app.retrieval.hybrid import RetrievalMode, RetrievedChunk, search_many

log = logging.getLogger(__name__)

# Cheap gate before spending a Gemini call: words that often join two separately-findable parts
_MAYBE_MULTI = re.compile(r"\b(and|versus|vs|compared?|comparison|differ\w*|both|each|respectively)\b|;|\?.*\?",
                          re.IGNORECASE)


class RAGAnswer(BaseModel):
    answer: str
    citations: list[cit.Citation]
    sources: list[RetrievedChunk]
    queries: list[str] | None = None  # what retrieval searched for, when it wasn't the question as asked
    model: str | None = None  # the Gemini model that wrote the answer


class _Plan(BaseModel):
    queries: list[str]


def plan_queries(question: str, history: list[dict] | None, multi_query: bool = True) -> list[str]:
    """The search queries for a question. A follow-up ("what are its limitations?") is rewritten to stand
    alone, and with multi_query a question about several things ("how do A and B differ?") is split, since
    one query embedding tends to settle on one part. On any failure, search the question as asked."""
    if not history and not (multi_query and _MAYBE_MULTI.search(question)):
        return [question]
    try:
        raw = get_llm().generate_json(plan_prompt(question, history), system=PLAN_SYSTEM, schema=_Plan,
                                      temperature=0.0, quick=True)
        planned = _Plan.model_validate(raw).queries
    except LLMNotConfigured:
        raise
    except Exception as e:  # noqa: BLE001 - a failed plan shouldn't block the answer
        log.warning("Query planning failed (%s); searching the question as asked", e)
        return [question]
    queries = [q.strip().strip('"“”').strip() for q in planned]
    queries = list(dict.fromkeys(q for q in queries if 3 <= len(q) <= 400))
    if not multi_query:
        queries = queries[:1]
    return queries[:MAX_SUBQUERIES] or [question]


def _finalize(raw: str, sources: list[RetrievedChunk], queries: list[str] | None, model: str | None) -> RAGAnswer:
    text = cit.strip_invalid(raw.strip(), len(sources))
    return RAGAnswer(answer=text, citations=cit.resolve(text, sources), sources=sources, queries=queries, model=model)


def answer(
    question: str,
    filters: SearchFilters | None = None,
    history: list[dict] | None = None,
    mode: RetrievalMode = RetrievalMode.hybrid,
    rerank: bool = False,
    top_k: int | None = None,
    multi_query: bool = True,
) -> RAGAnswer:
    queries = plan_queries(question, history, multi_query)
    shown = queries if queries != [question] else None
    sources = search_many(queries, filters=filters, mode=mode, rerank=rerank, top_k=top_k)
    if not sources:
        return RAGAnswer(answer=NOT_FOUND, citations=[], sources=[], queries=shown)
    info: dict = {}
    raw = get_llm().generate(answer_prompt(question, sources, history), system=ANSWER_SYSTEM, info=info)
    return _finalize(raw, sources, shown, info.get("model"))


def answer_stream(
    question: str,
    filters: SearchFilters | None = None,
    history: list[dict] | None = None,
    mode: RetrievalMode = RetrievalMode.hybrid,
    rerank: bool = False,
    top_k: int | None = None,
    multi_query: bool = True,
) -> Iterator[tuple[str, object]]:
    """Yields ("queries", [...]) if retrieval searched something other than the question as asked,
    then ("sources", [...]), ("token", str)*, and finally ("done", RAGAnswer)."""
    queries = plan_queries(question, history, multi_query)
    shown = queries if queries != [question] else None
    if shown:
        yield "queries", shown
    sources = search_many(queries, filters=filters, mode=mode, rerank=rerank, top_k=top_k)
    yield "sources", [s.model_dump() for s in sources]
    if not sources:
        yield "token", NOT_FOUND
        yield "done", RAGAnswer(answer=NOT_FOUND, citations=[], sources=[], queries=shown)
        return
    info: dict = {}
    parts: list[str] = []
    for token in get_llm().stream(answer_prompt(question, sources, history), system=ANSWER_SYSTEM, info=info):
        parts.append(token)
        yield "token", token
    yield "done", _finalize("".join(parts), sources, shown, info.get("model"))
