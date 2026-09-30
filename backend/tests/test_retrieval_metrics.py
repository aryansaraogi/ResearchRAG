import math

import pytest

from app.evaluation.retrieval_metrics import (
    hit_at_k, mean_scores, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank,
)

RANKED = ["a", "b", "c", "d", "e"]


def test_hit_and_recall():
    assert hit_at_k(RANKED, {"c"}, 2) == 0.0
    assert hit_at_k(RANKED, {"c"}, 3) == 1.0
    assert recall_at_k(RANKED, {"b", "z"}, 5) == 0.5
    assert recall_at_k(RANKED, set(), 5) == 0.0


def test_precision():
    assert precision_at_k(RANKED, {"a", "b"}, 4) == 0.5


def test_mrr():
    assert reciprocal_rank(RANKED, {"c"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(RANKED, {"z"}) == 0.0


def test_ndcg():
    assert ndcg_at_k(RANKED, {"a"}, 5) == 1.0
    assert ndcg_at_k(RANKED, {"b"}, 5) == pytest.approx(1 / math.log2(3))
    assert ndcg_at_k(RANKED, {"z"}, 5) == 0.0


def test_mean_scores_skips_none():
    rows = [{"x": 1.0, "y": None}, {"x": 0.0, "y": 1.0}]
    assert mean_scores(rows) == {"x": 0.5, "y": 1.0}
