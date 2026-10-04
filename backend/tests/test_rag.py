from app.generation import rag
from app.retrieval.hybrid import RetrievedChunk, rrf_merge

HISTORY = [
    {"role": "user", "content": "What is BERT?"},
    {"role": "assistant", "content": "BERT is a bidirectional Transformer encoder [1]."},
]


class _FakeLLM:
    def __init__(self, queries=None, error: Exception | None = None):
        self.queries, self.error, self.prompts = queries, error, []

    def generate_json(self, prompt, system=None, schema=None, temperature=0.0, info=None, *, quick=False):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return {"queries": self.queries}


def _use(monkeypatch, fake):
    monkeypatch.setattr(rag, "get_llm", lambda: fake)
    return fake


def test_simple_first_question_skips_the_planning_call(monkeypatch):
    fake = _use(monkeypatch, _FakeLLM(["unused"]))
    assert rag.plan_queries("What is masked language modeling?", []) == ["What is masked language modeling?"]
    assert fake.prompts == []


def test_follow_up_is_rewritten_with_the_conversation(monkeypatch):
    fake = _use(monkeypatch, _FakeLLM(['"What limitations of BERT do its authors acknowledge?"']))
    assert rag.plan_queries("What are its limitations?", HISTORY) == ["What limitations of BERT do its authors acknowledge?"]
    assert "What is BERT?" in fake.prompts[0]


def test_multi_part_question_is_split(monkeypatch):
    _use(monkeypatch, _FakeLLM(["BERT pre-training Adam settings", "Transformer Adam settings", "BERT pre-training Adam settings"]))
    question = "How do the Adam settings for BERT and the Transformer differ?"
    assert rag.plan_queries(question, None) == ["BERT pre-training Adam settings", "Transformer Adam settings"]
    # With splitting off, the same plan collapses to its first query
    assert rag.plan_queries(question, HISTORY, multi_query=False) == ["BERT pre-training Adam settings"]


def test_plan_is_capped(monkeypatch):
    _use(monkeypatch, _FakeLLM([f"query {i}" for i in range(6)]))
    assert len(rag.plan_queries("Compare A and B and C and D", None)) == 3


def test_failed_or_empty_plan_falls_back_to_the_question(monkeypatch):
    for fake in (_FakeLLM(error=RuntimeError("503")), _FakeLLM([]), _FakeLLM(["x" * 500])):
        _use(monkeypatch, fake)
        assert rag.plan_queries("What are its limitations?", HISTORY) == ["What are its limitations?"]


def _chunk(cid: str) -> RetrievedChunk:
    return RetrievedChunk(id=cid, paper_id="p", title="t", authors=[], year=None, section="s", section_type="other",
                          page_start=1, page_end=1, text="", score=0.0)


def test_rrf_merge_interleaves_lists_and_boosts_shared_hits():
    a = [_chunk("a1"), _chunk("shared"), _chunk("a3")]
    b = [_chunk("b1"), _chunk("shared"), _chunk("b3")]
    merged = [c.id for c in rrf_merge([a, b], limit=4)]
    # Found by both lists, "shared" outranks either list's first hit; then each list's top hits follow
    assert merged == ["shared", "a1", "b1", "a3"]
