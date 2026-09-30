"""Retrieval pipeline: dense / sparse / hybrid (RRF) search, optional cross-encoder rerank."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel
from qdrant_client import models

from app.config import get_settings
from app.retrieval import embeddings, reranker
from app.retrieval.filters import SearchFilters, to_qdrant_filter
from app.retrieval.vector_store import DENSE, SPARSE, client, collection_name


class RetrievalMode(str, Enum):
    dense = "dense"
    sparse = "sparse"
    hybrid = "hybrid"


class RetrievedChunk(BaseModel):
    id: str
    paper_id: str
    title: str
    authors: list[str]
    year: int | None
    section: str
    section_type: str
    page_start: int
    page_end: int
    text: str
    score: float  # fused/vector score
    rerank_score: float | None = None


def _to_chunk(p: models.ScoredPoint) -> RetrievedChunk:
    pl = p.payload or {}
    return RetrievedChunk(
        id=str(p.id),
        paper_id=pl["paper_id"],
        title=pl.get("title", ""),
        authors=pl.get("authors", []),
        year=pl.get("year"),
        section=pl.get("section", ""),
        section_type=pl.get("section_type", "other"),
        page_start=pl.get("page_start", 0),
        page_end=pl.get("page_end", 0),
        text=pl.get("text", ""),
        score=p.score,
    )


def search(
    query: str,
    filters: SearchFilters | None = None,
    mode: RetrievalMode = RetrievalMode.hybrid,
    rerank: bool = True,
    top_k: int | None = None,
    prefetch_k: int | None = None,
) -> list[RetrievedChunk]:
    s = get_settings()
    top_k = top_k or s.final_k
    prefetch_k = prefetch_k or s.prefetch_k
    flt = to_qdrant_filter(filters)
    # Without reranking, fetch exactly top_k; with it, fetch a wider candidate pool.
    limit = max(s.rerank_candidates, top_k) if rerank else top_k
    qc, name = client(), collection_name()

    if mode == RetrievalMode.dense:
        res = qc.query_points(name, query=embeddings.embed_dense_queries([query])[0], using=DENSE,
                              query_filter=flt, limit=limit, with_payload=True)
    elif mode == RetrievalMode.sparse:
        res = qc.query_points(name, query=embeddings.embed_sparse_query(query), using=SPARSE,
                              query_filter=flt, limit=limit, with_payload=True)
    else:
        res = qc.query_points(
            name,
            prefetch=[
                models.Prefetch(query=embeddings.embed_dense_queries([query])[0], using=DENSE,
                                filter=flt, limit=prefetch_k),
                models.Prefetch(query=embeddings.embed_sparse_query(query), using=SPARSE,
                                filter=flt, limit=prefetch_k),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )

    chunks = [_to_chunk(p) for p in res.points]
    if rerank and chunks:
        scores = reranker.rerank(query, [f"{c.title}. {c.section}. {c.text}" for c in chunks])
        for c, sc in zip(chunks, scores):
            c.rerank_score = float(sc)
        chunks.sort(key=lambda c: c.rerank_score, reverse=True)
    return chunks[:top_k]
