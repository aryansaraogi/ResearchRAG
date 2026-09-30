"""Section-aware chunking: chunks never cross section boundaries, carry page
ranges for citations, and overlap slightly so context isn't lost at edges."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from app.ingestion.pdf_parser import ParsedDocument, is_reference_section, section_type

WORDS_PER_TOKEN = 0.75
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


@dataclass
class Chunk:
    id: str
    paper_id: str
    chunk_index: int
    section: str
    section_type: str
    page_start: int
    page_end: int
    text: str


def chunk_id(paper_id: str, index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"researchrag/{paper_id}/{index}"))


def _units(text: str, page: int, max_words: int) -> list[tuple[list[str], int]]:
    """Split a paragraph into sentences (word lists), hard-splitting any longer than max_words."""
    units = []
    for sentence in _SENTENCE_SPLIT.split(text):
        sw = sentence.split()
        while len(sw) > max_words:  # pathological very long "sentence" (tables, equations)
            units.append((sw[:max_words], page))
            sw = sw[max_words:]
        if sw:
            units.append((sw, page))
    return units


def chunk_document(
    doc: ParsedDocument,
    paper_id: str,
    chunk_tokens: int = 400,
    overlap: float = 0.15,
    skip_references: bool = True,
) -> list[Chunk]:
    max_words = max(50, int(chunk_tokens * WORDS_PER_TOKEN))
    overlap_words = int(max_words * overlap)
    chunks: list[Chunk] = []

    for section in doc.sections:
        if skip_references and is_reference_section(section.title):
            continue
        kind = section_type(section.title)
        units = [u for p in section.paragraphs for u in _units(p.text, p.page, max_words)]

        # (word, page) stream so overlaps keep correct page attribution
        window: list[tuple[str, int]] = []

        def emit() -> None:
            if not window:
                return
            chunks.append(
                Chunk(
                    id=chunk_id(paper_id, len(chunks)),
                    paper_id=paper_id,
                    chunk_index=len(chunks),
                    section=section.title,
                    section_type=kind,
                    page_start=min(p for _, p in window),
                    page_end=max(p for _, p in window),
                    text=" ".join(w for w, _ in window),
                )
            )

        fresh = 0  # words added since last emit (excludes carried overlap)
        for words, page in units:
            if fresh and len(window) + len(words) > max_words:
                emit()
                keep = min(overlap_words, max_words - len(words))  # overlap must leave room
                window = window[-keep:] if keep > 0 else []
                fresh = 0
            window.extend((w, page) for w in words)
            fresh += len(words)
        if fresh:
            emit()

    # Drop tiny fragments (captions, stray lines), then re-index contiguously
    kept = [c for c in chunks if len(c.text.split()) >= 12 or c.section_type == "abstract"]
    for i, c in enumerate(kept):
        c.chunk_index = i
        c.id = chunk_id(paper_id, i)
    return kept
