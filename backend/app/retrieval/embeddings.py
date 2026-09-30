"""Local embedding models via FastEmbed (ONNX, no torch, no API rate limits)."""

from __future__ import annotations

from functools import lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import models

from app.config import get_settings

# BGE models retrieve better when queries (not passages) carry this instruction.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache
def dense_model() -> TextEmbedding:
    s = get_settings()
    return TextEmbedding(s.dense_model, cache_dir=str(s.model_cache_dir))


@lru_cache
def sparse_model() -> SparseTextEmbedding:
    s = get_settings()
    return SparseTextEmbedding(s.sparse_model, cache_dir=str(s.model_cache_dir))


@lru_cache
def cross_encoder() -> TextCrossEncoder:
    s = get_settings()
    return TextCrossEncoder(s.rerank_model, cache_dir=str(s.model_cache_dir))


def dense_dim() -> int:
    return len(embed_dense_queries(["dimension probe"])[0])


def embed_dense_passages(texts: list[str]) -> list[list[float]]:
    return [v.tolist() for v in dense_model().embed(texts, batch_size=32)]


def embed_dense_queries(texts: list[str]) -> list[list[float]]:
    if "bge" in get_settings().dense_model.lower():
        texts = [BGE_QUERY_PREFIX + t for t in texts]
    return [v.tolist() for v in dense_model().embed(texts)]


def embed_sparse_passages(texts: list[str]) -> list[models.SparseVector]:
    return [
        models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
        for e in sparse_model().embed(texts, batch_size=64)
    ]


def embed_sparse_query(text: str) -> models.SparseVector:
    e = next(iter(sparse_model().query_embed(text)))
    return models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
