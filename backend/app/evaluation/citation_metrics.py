"""Citation accuracy metrics computed from per-claim judge verdicts."""

from __future__ import annotations

from app.generation.citations import claims_with_citations


def citation_scores(answer: str, support: dict[tuple[int, int], bool], source_ids: list[str],
                    gold: set[str]) -> dict[str, float | None]:
    """
    support: {(claim_index, source_number): judged_supported}
    - citation_precision: share of (claim, cited source) pairs where the source supports the claim
    - citation_coverage:  share of answer sentences that carry at least one citation
    - gold_cited:         1 if the answer cites a gold chunk; None when no gold chunk was retrieved
    """
    claims = claims_with_citations(answer)
    pairs = [(i, n) for i, (_, nums) in enumerate(claims) for n in nums]
    precision = sum(1 for p in pairs if support.get(p, False)) / len(pairs) if pairs else 0.0
    coverage = sum(1 for _, nums in claims if nums) / len(claims) if claims else 0.0
    cited_ids = {source_ids[n - 1] for _, n in pairs if 1 <= n <= len(source_ids)}
    return {
        "citation_precision": precision,
        "citation_coverage": coverage,
        "gold_cited": (1.0 if cited_ids & gold else 0.0) if set(source_ids) & gold else None,
    }
