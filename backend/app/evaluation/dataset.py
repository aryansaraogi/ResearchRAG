"""Evaluation dataset: synthetic QA pairs generated from indexed chunks, each tied
to the chunk it was written from (the gold chunk). Hand-written items can be
appended to the same JSONL file."""

from __future__ import annotations

import json
import random
from pathlib import Path

from pydantic import BaseModel

from app.config import get_settings
from app.generation.llm import LLMClient
from app.retrieval import vector_store

SKIP_SECTION_TYPES = {"front_matter", "appendix"}


class EvalItem(BaseModel):
    id: str
    question: str
    reference_answer: str
    gold_chunk_ids: list[str]
    paper_id: str
    section: str = ""
    source: str = "synthetic"  # synthetic | manual
    # False for questions the library cannot answer: the right response is a refusal (no gold chunks)
    answerable: bool = True


class _GenQA(BaseModel):
    chunk_number: int
    question: str
    answer: str


GEN_SYSTEM = "You write evaluation questions for a search engine over research papers."
GEN_PROMPT = """For EACH numbered passage below, write one question that:
- can be answered from that passage alone, with a specific factual answer (a number, method, finding, definition, design choice);
- is self-contained: name the paper's method/model/topic instead of saying "this paper", "the authors" or "the passage";
- is phrased the way a researcher would search, and does not copy long phrases verbatim from the passage.
Also give a concise reference answer (1-2 sentences) taken from the passage.
Skip a passage (omit it) only if it has no substantive content (e.g. just a table of numbers or a caption).

{passages}"""


def dataset_path() -> Path:
    p = get_settings().data_dir / "eval" / "dataset.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_dataset(path: Path | None = None) -> list[EvalItem]:
    path = path or dataset_path()
    if not path.exists():
        return []
    return [EvalItem.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_dataset(items: list[EvalItem], path: Path | None = None) -> None:
    path = path or dataset_path()
    path.write_text("\n".join(i.model_dump_json() for i in items) + "\n", encoding="utf-8")


def _sample_chunks(n: int, paper_ids: list[str] | None, seed: int) -> list:
    records = [
        r for r in vector_store.scroll_chunks(paper_ids)
        if r.payload.get("section_type") not in SKIP_SECTION_TYPES and len(r.payload.get("text", "").split()) >= 60
    ]
    rng = random.Random(seed)
    by_paper: dict[str, list] = {}
    for r in records:
        by_paper.setdefault(r.payload["paper_id"], []).append(r)
    for lst in by_paper.values():
        rng.shuffle(lst)
    # Round-robin across papers so every paper is represented
    sampled, lists = [], list(by_paper.values())
    while len(sampled) < n and any(lists):
        for lst in lists:
            if lst and len(sampled) < n:
                sampled.append(lst.pop())
    return sampled


def generate_dataset(llm: LLMClient, n: int = 50, paper_ids: list[str] | None = None, seed: int = 42,
                     batch_size: int = 5, append: bool = False, progress=None) -> list[EvalItem]:
    chunks = _sample_chunks(n, paper_ids, seed)
    items: list[EvalItem] = load_dataset() if append else []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        passages = "\n\n".join(
            f"Passage {i} (paper: {r.payload['title']}; section: {r.payload['section']}):\n{r.payload['text']}"
            for i, r in enumerate(batch, 1)
        )
        raw = llm.generate_json(GEN_PROMPT.format(passages=passages), system=GEN_SYSTEM,
                                schema=list[_GenQA], temperature=0.4)
        for qa in (_GenQA.model_validate(x) for x in raw or []):
            if not 1 <= qa.chunk_number <= len(batch):
                continue
            r = batch[qa.chunk_number - 1]
            items.append(EvalItem(
                id=f"q{len(items) + 1:03d}", question=qa.question.strip(), reference_answer=qa.answer.strip(),
                gold_chunk_ids=[str(r.id)], paper_id=r.payload["paper_id"], section=r.payload["section"],
            ))
        if progress:
            progress(f"Generated {len(items)} questions ({min(start + batch_size, len(chunks))}/{len(chunks)} chunks)")
    save_dataset(items)
    return items
