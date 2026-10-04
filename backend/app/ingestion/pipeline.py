"""Ingestion pipeline: PDF -> sections -> chunks -> dense+sparse embeddings -> Qdrant."""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from pathlib import Path

import pymupdf
from sqlmodel import Session

from app.config import get_settings
from app.db import engine
from app.ingestion import arxiv_client
from app.ingestion.chunker import chunk_document
from app.generation.llm import LLMNotConfigured, get_llm
from app.generation.summaries import Passage, summarize
from app.ingestion.pdf_parser import parse_pdf
from app.ingestion.references import save_references
from app.models import Paper
from app.retrieval import vector_store

log = logging.getLogger(__name__)

# Ingestion is CPU-heavy (embedding); serialize it so concurrent uploads don't thrash.
_INGEST_LOCK = threading.Lock()

_ARXIV_IN_PDF = re.compile(r"arXiv:(\d{4}\.\d{4,5})(v\d+)?")


def file_paper_id(data: bytes) -> str:
    return "pdf_" + hashlib.sha1(data).hexdigest()[:16]


def detect_arxiv_id(pdf_path: Path) -> str | None:
    """Most arXiv PDFs carry 'arXiv:XXXX.XXXXXvN' in the first-page sidebar."""
    try:
        with pymupdf.open(pdf_path) as doc:
            m = _ARXIV_IN_PDF.search(doc[0].get_text())
            return m.group(1) if m else None
    except Exception:  # noqa: BLE001 - detection is best effort
        return None


def enrich_from_arxiv(paper: Paper) -> None:
    arxiv_id = detect_arxiv_id(Path(paper.pdf_path))
    if not arxiv_id:
        return
    try:
        [meta] = arxiv_client.fetch_by_ids([arxiv_id])
    except Exception as e:  # noqa: BLE001
        log.warning("arXiv lookup for %s failed: %s", arxiv_id, e)
        return
    paper.title = meta.title
    paper.authors = meta.authors
    paper.year = meta.year
    paper.categories = meta.categories
    paper.abstract = meta.abstract


def ingest(paper_id: str) -> None:
    """Parse, chunk, embed and index a paper already registered in the DB. Runs in the background."""
    with _INGEST_LOCK:
        _ingest(paper_id)


def _ingest(paper_id: str) -> None:
    s = get_settings()
    with Session(engine) as session:
        paper = session.get(Paper, paper_id)
        if paper is None:
            return
        paper.status = "processing"
        session.add(paper)
        session.commit()

        chunks = []
        try:
            doc = parse_pdf(paper.pdf_path)
            save_references(paper.id, doc)
            if paper.source == "upload":
                enrich_from_arxiv(paper)
                if not paper.title or paper.title == Path(paper.pdf_path).stem:
                    paper.title = doc.title or paper.title

            chunks = chunk_document(doc, paper.id, s.chunk_tokens, s.chunk_overlap)
            if not chunks:
                raise ValueError("No extractable text found (scanned PDF?)")

            vector_store.ensure_collection()
            vector_store.delete_paper(paper.id)  # idempotent re-ingest
            vector_store.upsert_chunks(
                chunks,
                {
                    "title": paper.title,
                    "authors": paper.authors,
                    "year": paper.year,
                    "categories": paper.categories,
                },
            )
            paper.num_pages = doc.num_pages
            paper.num_chunks = len(chunks)
            paper.sections = [sec.title for sec in doc.sections]
            paper.status = "ready"
            paper.error = None
        except Exception as e:  # noqa: BLE001
            log.exception("Ingestion failed for %s", paper_id)
            paper.status = "failed"
            paper.error = str(e)[:500]
        session.add(paper)
        session.commit()

        # Summarize after the paper is already searchable, so a slow or failed Gemini call never holds it up
        if paper.status == "ready":
            paper.summary = try_summary(paper.title, [Passage(c.section, c.section_type, c.page_start, c.text)
                                                      for c in chunks])
            session.add(paper)
            session.commit()


def try_summary(title: str, passages: list[Passage]) -> dict | None:
    try:
        return summarize(get_llm(), title, passages)
    except LLMNotConfigured:
        return None
    except Exception as e:  # noqa: BLE001 - the paper stays usable without a summary
        log.warning("Summary failed for %s: %s", title, e)
        return None


def register_arxiv(session: Session, meta: arxiv_client.ArxivPaper) -> Paper:
    existing = session.get(Paper, meta.arxiv_id)
    if existing and existing.status in ("ready", "processing"):
        return existing
    pdf_path = arxiv_client.download_pdf(meta, get_settings().pdf_dir)
    paper = existing or Paper(id=meta.arxiv_id, title=meta.title)
    paper.title = meta.title
    paper.authors = meta.authors
    paper.year = meta.year
    paper.categories = meta.categories
    paper.abstract = meta.abstract
    paper.source = "arxiv"
    paper.pdf_path = str(pdf_path)
    paper.status = "pending"
    session.add(paper)
    session.commit()
    session.refresh(paper)
    return paper
