from app.retrieval.filters import SearchFilters, to_qdrant_filter


def test_empty_filter_is_none():
    assert to_qdrant_filter(None) is None
    assert to_qdrant_filter(SearchFilters()) is None


def test_filter_conditions():
    f = to_qdrant_filter(SearchFilters(year_min=2019, categories=["cs.CL"], section_types=["method"]))
    keys = {c.key for c in f.must}
    assert keys == {"year", "categories", "section_type"}
    year = next(c for c in f.must if c.key == "year")
    assert year.range.gte == 2019 and year.range.lte is None
