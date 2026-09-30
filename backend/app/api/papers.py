from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlmodel import Session, select

from app.config import get_settings
from app.db import get_session
from app.ingestion import arxiv_client
from app.ingestion.pipeline import file_paper_id, ingest, register_arxiv
from app.models import Paper
from app.retrieval import vector_store

router = APIRouter(prefix="/papers", tags=["papers"])

MAX_UPLOAD_MB = 50
SECTION_TYPES = ["abstract", "introduction", "related_work", "method", "experiments",
                 "results", "discussion", "conclusion", "appendix", "other"]


class ArxivImportRequest(BaseModel):
    ids: list[str]


@router.get("", response_model=list[Paper])
def list_papers(session: Session = Depends(get_session)):
    return session.exec(select(Paper).order_by(Paper.created_at.desc())).all()


@router.get("/facets")
def facets(session: Session = Depends(get_session)):
    """Distinct metadata values for building filter controls."""
    papers = session.exec(select(Paper).where(Paper.status == "ready")).all()
    years = sorted({p.year for p in papers if p.year})
    return {
        "authors": sorted({a for p in papers for a in p.authors}),
        "categories": sorted({c for p in papers for c in p.categories}),
        "year_min": years[0] if years else None,
        "year_max": years[-1] if years else None,
        "papers": [{"id": p.id, "title": p.title} for p in papers],
        "section_types": SECTION_TYPES,
    }


@router.get("/arxiv/search")
def arxiv_search(q: str, max_results: int = 10):
    try:
        return arxiv_client.search(q, max_results=min(max_results, 25))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"arXiv search failed: {e}") from e


@router.post("/arxiv", response_model=list[Paper])
def import_arxiv(req: ArxivImportRequest, background: BackgroundTasks, session: Session = Depends(get_session)):
    try:
        metas = arxiv_client.fetch_by_ids(req.ids)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"arXiv lookup failed: {e}") from e
    if not metas:
        raise HTTPException(404, "No arXiv papers found for those ids")
    papers = []
    for meta in metas:
        try:
            paper = register_arxiv(session, meta)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, f"Downloading {meta.arxiv_id} failed: {e}") from e
        if paper.status == "pending":
            background.add_task(ingest, paper.id)
        papers.append(paper)
    return papers


@router.post("/upload", response_model=list[Paper])
async def upload(background: BackgroundTasks, files: list[UploadFile] = File(...),
                 session: Session = Depends(get_session)):
    papers = []
    for f in files:
        data = await f.read()
        if not data.startswith(b"%PDF"):
            raise HTTPException(400, f"{f.filename} is not a PDF")
        if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(413, f"{f.filename} exceeds {MAX_UPLOAD_MB} MB")
        pid = file_paper_id(data)
        existing = session.get(Paper, pid)
        if existing and existing.status in ("ready", "processing", "pending"):
            papers.append(existing)
            continue
        path = get_settings().pdf_dir / f"{pid}.pdf"
        path.write_bytes(data)
        paper = existing or Paper(id=pid, title=Path(f.filename or pid).stem)
        paper.pdf_path, paper.source, paper.status = str(path), "upload", "pending"
        session.add(paper)
        session.commit()
        session.refresh(paper)
        background.add_task(ingest, pid)
        papers.append(paper)
    return papers


@router.get("/{paper_id}", response_model=Paper)
def get_paper(paper_id: str, session: Session = Depends(get_session)):
    paper = session.get(Paper, paper_id)
    if not paper:
        raise HTTPException(404, "Paper not found")
    return paper


@router.get("/{paper_id}/pdf")
def get_pdf(paper_id: str, session: Session = Depends(get_session)):
    paper = session.get(Paper, paper_id)
    if not paper or not Path(paper.pdf_path).exists():
        raise HTTPException(404, "PDF not found")
    return FileResponse(paper.pdf_path, media_type="application/pdf", headers={"Content-Disposition": "inline"})


@router.post("/{paper_id}/reingest", response_model=Paper)
def reingest(paper_id: str, background: BackgroundTasks, session: Session = Depends(get_session)):
    paper = session.get(Paper, paper_id)
    if not paper:
        raise HTTPException(404, "Paper not found")
    paper.status = "pending"
    session.add(paper)
    session.commit()
    session.refresh(paper)
    background.add_task(ingest, paper_id)
    return paper


@router.delete("/{paper_id}", status_code=204)
def delete_paper(paper_id: str, session: Session = Depends(get_session)):
    paper = session.get(Paper, paper_id)
    if not paper:
        raise HTTPException(404, "Paper not found")
    try:
        vector_store.delete_paper(paper_id)
    except Exception:  # noqa: BLE001 - collection may not exist yet
        pass
    Path(paper.pdf_path).unlink(missing_ok=True)
    session.delete(paper)
    session.commit()
