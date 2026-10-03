from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.retrieval.filters import SearchFilters
from app.retrieval.hybrid import RetrievalMode, RetrievedChunk, search

router = APIRouter(tags=["search"])


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    filters: SearchFilters | None = None
    mode: RetrievalMode = RetrievalMode.hybrid
    rerank: bool = False  # cross-encoder adds ~4 s on CPU; see the README evaluation
    top_k: int = Field(default=10, ge=1, le=50)


@router.post("/search", response_model=list[RetrievedChunk])
def search_chunks(req: SearchRequest):
    return search(req.query, filters=req.filters, mode=req.mode, rerank=req.rerank, top_k=req.top_k)
