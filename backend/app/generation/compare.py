"""Side-by-side paper comparison: for each paper, retrieve passages per aspect from that paper only,
then one Gemini call fills every aspect with cited text."""

from __future__ import annotations

from pydantic import BaseModel

from app.generation import citations as cit
from app.generation.llm import get_llm
from app.generation.prompts import format_sources
from app.retrieval.filters import SearchFilters
from app.retrieval.hybrid import RetrievalMode, RetrievedChunk, search

# aspect -> what to search for in each paper
DEFAULT_ASPECTS: dict[str, str] = {
    "Problem": "the problem or task the paper addresses and why it matters",
    "Method": "the proposed method, model architecture or approach",
    "Data": "datasets, benchmarks and experimental setup",
    "Results": "main quantitative results and findings",
    "Limitations": "limitations, weaknesses or open problems",
}
PASSAGES_PER_ASPECT = 3
NOT_COVERED = "Not covered in the retrieved passages."

COMPARE_SYSTEM = "You extract comparable facts from one research paper, using only the numbered passages provided."


class _Cell(BaseModel):
    aspect: str
    text: str


class _Cells(BaseModel):
    cells: list[_Cell]


class CompareCell(BaseModel):
    text: str
    citations: list[cit.Citation]


class PaperColumn(BaseModel):
    paper_id: str
    title: str
    year: int | None
    cells: dict[str, CompareCell]
    sources: list[RetrievedChunk]
    model: str | None = None


def _passages(paper_id: str, aspects: dict[str, str]) -> list[RetrievedChunk]:
    seen: dict[str, RetrievedChunk] = {}
    only_this = SearchFilters(paper_ids=[paper_id])
    for name, description in aspects.items():
        query = f"{name}: {description}" if description != name else name
        for c in search(query, filters=only_this, mode=RetrievalMode.hybrid, rerank=False, top_k=PASSAGES_PER_ASPECT):
            seen.setdefault(c.id, c)
    return list(seen.values())


def compare_prompt(title: str, aspects: dict[str, str], sources: list[RetrievedChunk]) -> str:
    wanted = "\n".join(f"- {name}: {desc}" for name, desc in aspects.items())
    return (
        f"Paper: {title}\n\nPassages:\n\n{format_sources(sources)}\n\n"
        f"For each aspect below, write 1-3 concise sentences about THIS paper, with numbers where the passages give "
        f"them. Cite the passage numbers in square brackets after each sentence, e.g. [2]. If the passages don't "
        f'cover an aspect, write exactly "{NOT_COVERED}"\n\nAspects:\n{wanted}'
    )


def compare_paper(paper_id: str, title: str, year: int | None, aspects: dict[str, str]) -> PaperColumn:
    sources = _passages(paper_id, aspects)
    if not sources:
        empty = CompareCell(text=NOT_COVERED, citations=[])
        return PaperColumn(paper_id=paper_id, title=title, year=year, cells=dict.fromkeys(aspects, empty), sources=[])
    info: dict = {}
    raw = get_llm().generate_json(compare_prompt(title, aspects, sources), system=COMPARE_SYSTEM, schema=_Cells,
                                  temperature=0.1, info=info)
    filled = {c.aspect.strip(): c.text for c in _Cells.model_validate(raw).cells}
    cells = {}
    for name in aspects:
        text = cit.strip_invalid(filled.get(name, NOT_COVERED).strip(), len(sources)) or NOT_COVERED
        cells[name] = CompareCell(text=text, citations=cit.resolve(text, sources))
    return PaperColumn(paper_id=paper_id, title=title, year=year, cells=cells, sources=sources, model=info.get("model"))
