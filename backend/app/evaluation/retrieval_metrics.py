"""Standard IR metrics over ranked chunk ids vs. a set of gold (relevant) ids."""

from __future__ import annotations

import math
from collections.abc import Sequence


def hit_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    return 1.0 if any(r in gold for r in ranked[:k]) else 0.0


def recall_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & gold) / len(gold)


def precision_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(ranked[:k]) & gold) / k


def reciprocal_rank(ranked: Sequence[str], gold: set[str]) -> float:
    for i, r in enumerate(ranked, start=1):
        if r in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, r in enumerate(ranked[:k], start=1) if r in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def retrieval_scores(ranked: Sequence[str], gold: set[str], ks: Sequence[int] = (1, 3, 5, 10)) -> dict[str, float]:
    scores = {"mrr": reciprocal_rank(ranked, gold)}
    for k in ks:
        scores[f"hit@{k}"] = hit_at_k(ranked, gold, k)
        scores[f"recall@{k}"] = recall_at_k(ranked, gold, k)
        scores[f"ndcg@{k}"] = ndcg_at_k(ranked, gold, k)
    scores["precision@5"] = precision_at_k(ranked, gold, 5)
    return scores


def mean_scores(rows: list[dict[str, float | None]]) -> dict[str, float]:
    """Average each metric across rows, ignoring None (metric not applicable to that row)."""
    out: dict[str, float] = {}
    for k in dict.fromkeys(k for r in rows for k in r):
        vals = [r[k] for r in rows if r.get(k) is not None]
        if vals:
            out[k] = round(sum(vals) / len(vals), 4)
    return out
