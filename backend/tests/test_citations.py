from app.evaluation.citation_metrics import citation_scores
from app.generation.citations import claims_with_citations, cited_numbers, strip_invalid


def test_cited_numbers_formats():
    assert cited_numbers("A [2]. B [1, 3]. C [2][4].") == [2, 1, 3, 4]


def test_strip_invalid_removes_out_of_range():
    assert strip_invalid("X [1][9]. Y [2, 7].", 3) == "X [1]. Y [2]."


def test_claims_attach_trailing_citation():
    claims = claims_with_citations("Transformers use attention [1]. They drop recurrence. [2]")
    assert claims == [("Transformers use attention", [1]), ("They drop recurrence", [2])]


def test_citation_scores():
    answer = "Claim one [1]. Claim two [2]. Uncited claim."
    support = {(0, 1): True, (1, 2): False}
    s = citation_scores(answer, support, ["c1", "c2"], gold={"c1"})
    assert s["citation_precision"] == 0.5
    assert abs(s["citation_coverage"] - 2 / 3) < 1e-9
    assert s["gold_cited"] == 1.0


def test_gold_cited_none_when_gold_not_retrieved():
    s = citation_scores("Claim [1].", {(0, 1): True}, ["c1"], gold={"other"})
    assert s["gold_cited"] is None
