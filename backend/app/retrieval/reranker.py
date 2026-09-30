"""Cross-encoder reranking of retrieved candidates."""

from __future__ import annotations

from app.retrieval import embeddings


def rerank(query: str, texts: list[str]) -> list[float]:
    if not texts:
        return []
    return list(embeddings.cross_encoder().rerank(query, texts, batch_size=16))
