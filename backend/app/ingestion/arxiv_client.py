"""arXiv search and import: metadata plus PDF download."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import arxiv
import httpx

_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?|([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?")


@dataclass
class ArxivPaper:
    arxiv_id: str  # without version suffix
    title: str
    authors: list[str]
    year: int
    categories: list[str]
    abstract: str
    pdf_url: str


def normalize_id(raw: str) -> str:
    """Accept '1706.03762', '1706.03762v7', 'arXiv:1706.03762' or an abs/pdf URL."""
    m = _ID_RE.search(raw.strip())
    if not m:
        raise ValueError(f"Not a recognizable arXiv id: {raw!r}")
    return m.group(1) or m.group(3)


def _to_paper(r: arxiv.Result) -> ArxivPaper:
    return ArxivPaper(
        arxiv_id=normalize_id(r.get_short_id()),
        title=" ".join(r.title.split()),
        authors=[a.name for a in r.authors],
        year=r.published.year,
        categories=list(r.categories),
        abstract=" ".join(r.summary.split()),
        pdf_url=r.pdf_url,
    )


_client = arxiv.Client(page_size=50, delay_seconds=3, num_retries=3)


def fetch_by_ids(ids: list[str]) -> list[ArxivPaper]:
    search = arxiv.Search(id_list=[normalize_id(i) for i in ids])
    return [_to_paper(r) for r in _client.results(search)]


def search(query: str, max_results: int = 10) -> list[ArxivPaper]:
    s = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    return [_to_paper(r) for r in _client.results(s)]


def download_pdf(paper: ArxivPaper, dest_dir: Path) -> Path:
    dest = dest_dir / f"arxiv_{paper.arxiv_id.replace('/', '_')}.pdf"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        resp = client.get(paper.pdf_url)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    return dest
