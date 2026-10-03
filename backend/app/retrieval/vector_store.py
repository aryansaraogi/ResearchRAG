"""Qdrant collection management: one collection, named dense + sparse vectors.

Runs against a Qdrant server when QDRANT_URL is set, otherwise uses Qdrant's
embedded local mode (same API, stored under data/qdrant_local, single process).
"""

from __future__ import annotations

import atexit
import threading
from functools import lru_cache
from typing import Any

from qdrant_client import QdrantClient, models

from app.config import get_settings
from app.ingestion.chunker import Chunk
from app.retrieval import embeddings

DENSE = "dense"
SPARSE = "sparse"

KEYWORD_INDEXES = ("paper_id", "authors", "categories", "section_type")


class _SerializedClient:
    """Embedded Qdrant isn't thread-safe; serialize calls from API and ingestion threads."""

    def __init__(self, inner: QdrantClient):
        self._inner = inner
        self._lock = threading.RLock()

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        def locked(*args: Any, **kwargs: Any) -> Any:
            with self._lock:
                return attr(*args, **kwargs)

        return locked


@lru_cache
def client() -> QdrantClient:
    s = get_settings()
    if s.qdrant_url:
        return QdrantClient(url=s.qdrant_url, timeout=60)
    local = QdrantClient(path=str(s.qdrant_local_path))
    atexit.register(local.close)  # flush and release the storage lock cleanly on exit
    return _SerializedClient(local)  # type: ignore[return-value]


def collection_name() -> str:
    return get_settings().qdrant_collection


def ensure_collection() -> None:
    qc = client()
    name = collection_name()
    if qc.collection_exists(name):
        return
    qc.create_collection(
        name,
        vectors_config={DENSE: models.VectorParams(size=embeddings.dense_dim(), distance=models.Distance.COSINE)},
        # BM25 term weights need IDF computed across the collection at query time
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    if get_settings().qdrant_url:  # payload indexes only exist on the server
        for field in KEYWORD_INDEXES:
            qc.create_payload_index(name, field, models.PayloadSchemaType.KEYWORD)
        qc.create_payload_index(name, "year", models.PayloadSchemaType.INTEGER)


def embedding_text(chunk: Chunk, title: str) -> str:
    """Prefix each chunk with its paper and section so short chunks keep their context."""
    return f"{title}\nSection: {chunk.section}\n{chunk.text}"


def upsert_chunks(chunks: list[Chunk], paper_meta: dict, batch_size: int = 64) -> None:
    qc = client()
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [embedding_text(c, paper_meta["title"]) for c in batch]
        dense = embeddings.embed_dense_passages(texts)
        sparse = embeddings.embed_sparse_passages(texts)
        points = [
            models.PointStruct(
                id=c.id,
                vector={DENSE: d, SPARSE: sp},
                payload={
                    **paper_meta,
                    "paper_id": c.paper_id,
                    "chunk_index": c.chunk_index,
                    "section": c.section,
                    "section_type": c.section_type,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "text": c.text,
                },
            )
            for c, d, sp in zip(batch, dense, sparse)
        ]
        qc.upsert(collection_name(), points=points, wait=True)


def delete_paper(paper_id: str) -> None:
    client().delete(
        collection_name(),
        points_selector=models.FilterSelector(
            filter=models.Filter(must=[models.FieldCondition(key="paper_id", match=models.MatchValue(value=paper_id))])
        ),
        wait=True,
    )


def get_chunks(ids: list[str]) -> list[models.Record]:
    """Fetch chunk payloads by id; ids that do not exist are skipped."""
    if not ids:
        return []
    return client().retrieve(collection_name(), ids=ids, with_payload=True, with_vectors=False)


def scroll_chunks(paper_ids: list[str] | None = None, limit: int = 10_000) -> list[models.Record]:
    flt = None
    if paper_ids:
        flt = models.Filter(must=[models.FieldCondition(key="paper_id", match=models.MatchAny(any=paper_ids))])
    records, offset = [], None
    while len(records) < limit:
        page, offset = client().scroll(
            collection_name(), scroll_filter=flt, limit=256, offset=offset, with_payload=True, with_vectors=False
        )
        records.extend(page)
        if offset is None:
            break
    return records[:limit]
