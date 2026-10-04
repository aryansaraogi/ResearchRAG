"""Paper summaries made at import: one grounded Gemini call per paper (TL;DR, contributions, limitations)."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from app.generation.llm import LLMClient

MAX_WORDS = 3500
# Where papers state what they did and what is left: read these first, the introduction last (it is long)
PRIORITY = ("abstract", "conclusion", "discussion", "introduction")

SUMMARY_SYSTEM = "You summarize research papers for researchers, using only the passages provided."


@dataclass
class Passage:
    section: str
    section_type: str
    page: int
    text: str


class SummaryPoint(BaseModel):
    text: str
    section: str  # heading of the passage the point comes from


class PaperSummary(BaseModel):
    tldr: str
    contributions: list[SummaryPoint]
    limitations: list[SummaryPoint]


def select_passages(passages: list[Passage], max_words: int = MAX_WORDS) -> list[Passage]:
    """Abstract, conclusion/discussion (plus any section titled "limitations"), then introduction, within budget.
    Papers whose headings weren't recognised fall back to their opening passages."""

    def rank(p: Passage) -> int:
        if "limitation" in p.section.lower():
            return 1
        return PRIORITY.index(p.section_type) if p.section_type in PRIORITY else len(PRIORITY)

    candidates = [p for p in passages if rank(p) < len(PRIORITY)] or passages
    chosen, words = [], 0
    for p in sorted(candidates, key=rank):
        n = len(p.text.split())
        if words + n > max_words and chosen:
            continue
        chosen.append(p)
        words += n
    return [p for p in passages if p in chosen]  # back in reading order


def summary_prompt(title: str, passages: list[Passage]) -> str:
    body = "\n\n".join(f"[§ {p.section}, p. {p.page}]\n{p.text}" for p in passages)
    return (
        f"Paper: {title}\n\nPassages:\n\n{body}\n\n"
        "Write:\n"
        "- tldr: 1-2 sentences on what the paper does and its main result.\n"
        "- contributions: 3-5 key contributions, one sentence each.\n"
        "- limitations: up to 4 limitations or open problems that the passages state or clearly imply. "
        "If they state none, return an empty list rather than inventing one.\n"
        "For every contribution and limitation, set section to the heading of the passage it comes from, "
        "exactly as written after the § sign."
    )


def summarize(llm: LLMClient, title: str, passages: list[Passage]) -> dict:
    chosen = select_passages(passages)
    info: dict = {}
    raw = llm.generate_json(summary_prompt(title, chosen), system=SUMMARY_SYSTEM, schema=PaperSummary,
                            temperature=0.2, info=info)
    summary = PaperSummary.model_validate(raw)
    # Drop section labels that don't match a passage we sent (the model must not invent locations)
    known = {p.section for p in chosen}
    for point in summary.contributions + summary.limitations:
        if point.section not in known:
            point.section = ""
    return {**summary.model_dump(), "model": info.get("model")}
