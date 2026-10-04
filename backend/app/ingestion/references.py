"""Reference lists and the in-library citation graph ("A cites B" between indexed papers).

The parser already isolates each paper's References section (the chunker skips it), so the list is saved
at import, one entry per line, under data/refs/. Matching is deterministic, with no LLM calls: a paper cites
a library paper if its reference list contains that paper's arXiv id or its normalised title.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

from app.config import get_settings
from app.ingestion.pdf_parser import ParsedDocument, is_reference_section, parse_pdf
from app.models import Paper

log = logging.getLogger(__name__)

MIN_TITLE_WORDS = 4  # shorter titles are too generic to match safely
TITLE_PREFIX_WORDS = 8  # long titles get line-broken or abbreviated in reference lists


def refs_path(paper_id: str) -> Path:
    d = get_settings().data_dir / "refs"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{re.sub(r'[^A-Za-z0-9._-]', '_', paper_id)}.txt"


def save_references(paper_id: str, doc: ParsedDocument) -> str:
    entries = [p.text.strip() for s in doc.sections if is_reference_section(s.title) for p in s.paragraphs]
    text = "\n".join(e for e in entries if e)
    refs_path(paper_id).write_text(text, encoding="utf-8")
    return text


def load_references(paper: Paper) -> str:
    """Saved reference list; papers imported before this existed are parsed once, on first use."""
    path = refs_path(paper.id)
    if path.exists():
        return path.read_text(encoding="utf-8")
    if not paper.pdf_path or not Path(paper.pdf_path).exists():
        return ""
    try:
        return save_references(paper.id, parse_pdf(paper.pdf_path))
    except Exception as e:  # noqa: BLE001 - a broken PDF just has no references
        log.warning("Couldn't extract references from %s: %s", paper.id, e)
        return ""


def normalize(text: str) -> str:
    """Lowercase ASCII words only, so "Pre-\\ntraining" and "Pre-training" compare equal."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    # Join every hyphen, both in-word ("Pre-training") and across a line break ("Pre-\ntraining")
    text = re.sub(r"-\s*", "", text.lower())
    return " ".join(re.findall(r"[a-z0-9]+", text))


def cites(references: str, target: Paper) -> bool:
    if not references:
        return False
    if target.source == "arxiv" and re.search(rf"(?<![\d.]){re.escape(target.id)}(?!\d)", references):
        return True
    title = normalize(target.title).split()
    if len(title) < MIN_TITLE_WORDS:
        return False
    refs = normalize(references)
    return " ".join(title) in refs or (
        len(title) > TITLE_PREFIX_WORDS and " ".join(title[:TITLE_PREFIX_WORDS]) in refs
    )


def count_references(text: str) -> int:
    """Entries in a reference list. Entries wrap over several lines, so count by citation style: the highest
    "[n]" marker for numbered lists, otherwise one "2018." (author-year) per entry. Approximate for the latter."""
    markers = [int(n) for n in re.findall(r"^\s*\[(\d+)\]", text, re.MULTILINE)]
    if len(markers) >= 3:
        return max(markers)
    return len(re.findall(r"(?<![\d.])(?:19|20)\d{2}[a-z]?\.(?=\s|$)", text, re.MULTILINE))


def citation_graph(papers: list[Paper]) -> dict:
    refs = {p.id: load_references(p) for p in papers}
    edges = [
        {"source": a.id, "target": b.id}
        for a in papers
        for b in papers
        if a.id != b.id and cites(refs[a.id], b)
    ]
    nodes = [
        {"id": p.id, "title": p.title, "year": p.year, "authors": p.authors,
         "n_references": count_references(refs[p.id])}
        for p in papers
    ]
    return {"nodes": nodes, "edges": edges}
