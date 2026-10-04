from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session

from app.db import get_session
from app.generation.compare import DEFAULT_ASPECTS, PaperColumn, compare_paper
from app.generation.llm import LLMNotConfigured, describe_error
from app.models import Paper

router = APIRouter(prefix="/compare", tags=["compare"])

MAX_ASPECTS = 8


class CompareRequest(BaseModel):
    paper_ids: list[str] = Field(min_length=2, max_length=4)
    # Aspect names; known ones (Problem, Method, ...) carry a search description, custom ones are searched as written
    aspects: list[str] | None = None


class CompareResponse(BaseModel):
    aspects: list[str]
    papers: list[PaperColumn]


@router.get("/aspects")
def default_aspects() -> dict[str, str]:
    return DEFAULT_ASPECTS


@router.post("", response_model=CompareResponse)
def compare(req: CompareRequest, session: Session = Depends(get_session)):
    names = list(dict.fromkeys(a.strip()[:80] for a in (req.aspects or DEFAULT_ASPECTS) if a.strip()))[:MAX_ASPECTS]
    if not names:
        raise HTTPException(400, "Pick at least one aspect to compare")
    aspects = {n: DEFAULT_ASPECTS.get(n, n) for n in names}
    papers = []
    for pid in dict.fromkeys(req.paper_ids):
        paper = session.get(Paper, pid)
        if not paper:
            raise HTTPException(404, f"Paper {pid} not found")
        if paper.status != "ready":
            raise HTTPException(409, f"{paper.title} is still being indexed")
        papers.append(paper)
    try:
        columns = [compare_paper(p.id, p.title, p.year, aspects) for p in papers]
    except LLMNotConfigured as e:
        raise HTTPException(503, str(e)) from e
    except Exception as e:  # noqa: BLE001 - Gemini errors become a readable message
        raise HTTPException(502, describe_error(e)) from e
    return CompareResponse(aspects=names, papers=columns)
