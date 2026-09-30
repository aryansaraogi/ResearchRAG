"""Parse [n] citation markers from generated answers and map them to source chunks."""

from __future__ import annotations

import re

from pydantic import BaseModel

from app.retrieval.hybrid import RetrievedChunk

_CITATION = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


class Citation(BaseModel):
    number: int
    chunk_id: str
    paper_id: str
    title: str
    section: str
    page_start: int
    page_end: int
    snippet: str


def cited_numbers(text: str) -> list[int]:
    """All citation numbers in order of first appearance, handling [1], [1, 2] and [1][2]."""
    seen: list[int] = []
    for m in _CITATION.finditer(text):
        for n in re.split(r"\s*[,;]\s*", m.group(1)):
            k = int(n)
            if k not in seen:
                seen.append(k)
    return seen


def strip_invalid(text: str, n_sources: int) -> str:
    """Remove citation markers that point outside the provided sources (hallucinated ids)."""

    def fix(m: re.Match) -> str:
        valid = [n for n in re.split(r"\s*[,;]\s*", m.group(1)) if 1 <= int(n) <= n_sources]
        return "".join(f"[{n}]" for n in valid)

    return _CITATION.sub(fix, text)


def resolve(text: str, chunks: list[RetrievedChunk]) -> list[Citation]:
    out = []
    for n in cited_numbers(text):
        if 1 <= n <= len(chunks):
            c = chunks[n - 1]
            out.append(Citation(number=n, chunk_id=c.id, paper_id=c.paper_id, title=c.title,
                                section=c.section, page_start=c.page_start, page_end=c.page_end,
                                snippet=c.text[:400]))
    return out


def claims_with_citations(text: str) -> list[tuple[str, list[int]]]:
    """Split an answer into sentences, returning each with the citation numbers it carries."""
    out: list[tuple[str, list[int]]] = []
    for sent in _SENTENCE.split(text):
        sent = sent.strip()
        nums = cited_numbers(sent)
        clean = _CITATION.sub("", sent).strip(" .,;")
        if len(clean) >= 3:
            out.append((clean, nums))
        elif nums and out:  # "... claim. [1]" — citation split off after the period
            prev, prev_nums = out[-1]
            out[-1] = (prev, prev_nums + [n for n in nums if n not in prev_nums])
    return out
