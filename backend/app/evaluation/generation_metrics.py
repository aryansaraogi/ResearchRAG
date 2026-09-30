"""LLM-as-judge generation metrics. One judge call per answer scores the answer
and every (claim, cited source) pair, to stay inside free-tier quotas."""

from __future__ import annotations

from pydantic import BaseModel

from app.generation.citations import claims_with_citations
from app.generation.llm import LLMClient
from app.retrieval.hybrid import RetrievedChunk

JUDGE_SYSTEM = """You are a strict evaluator of retrieval-augmented answers about research papers.
Score using only the provided material. Be critical; do not reward fluent but unsupported content."""


class ClaimVerdict(BaseModel):
    claim_index: int
    source_number: int
    supported: bool


class JudgeResult(BaseModel):
    faithfulness: int  # 1-5: every claim is supported by the sources
    answer_relevance: int  # 1-5: directly and completely addresses the question
    context_relevance: int  # 1-5: retrieved sources are relevant to the question
    correctness: int  # 1-5: agrees with the reference answer
    claim_verdicts: list[ClaimVerdict]


RUBRIC = """Score each dimension from 1 (very poor) to 5 (excellent):
- faithfulness: are ALL claims in the answer supported by the sources? (5 = fully grounded, 1 = mostly unsupported)
- answer_relevance: does the answer directly and completely address the question?
- context_relevance: how relevant are the retrieved sources to the question overall?
- correctness: does the answer agree with the reference answer? (a "not found" answer scores 1 unless the reference is also absent)
Then, for EVERY (claim, cited source) pair listed under "Citations to verify", set supported=true only if
that specific source states or directly implies that claim."""


def _fmt_sources(sources: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{i}] ({s.title} — §{s.section})\n{s.text}" for i, s in enumerate(sources, 1))


def judge(llm: LLMClient, question: str, reference: str, answer: str,
          sources: list[RetrievedChunk]) -> tuple[dict[str, float], dict[tuple[int, int], bool]]:
    claims = claims_with_citations(answer)
    pairs = "\n".join(
        f"- claim {i}, source [{n}]: {text}" for i, (text, nums) in enumerate(claims) for n in nums
    ) or "(none)"
    prompt = (
        f"{RUBRIC}\n\nQuestion: {question}\n\nReference answer: {reference}\n\n"
        f"Sources:\n{_fmt_sources(sources)}\n\nAnswer under evaluation:\n{answer}\n\n"
        f"Citations to verify:\n{pairs}"
    )
    raw = llm.generate_json(prompt, system=JUDGE_SYSTEM, schema=JudgeResult)
    result = JudgeResult.model_validate(raw)

    def norm(x: int) -> float:
        return round((min(max(x, 1), 5) - 1) / 4, 4)  # 1..5 -> 0..1

    scores = {
        "faithfulness": norm(result.faithfulness),
        "answer_relevance": norm(result.answer_relevance),
        "context_relevance": norm(result.context_relevance),
        "correctness": norm(result.correctness),
    }
    support = {(v.claim_index, v.source_number): v.supported for v in result.claim_verdicts}
    return scores, support
