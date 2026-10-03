from app.generation import rag

HISTORY = [
    {"role": "user", "content": "What is BERT?"},
    {"role": "assistant", "content": "BERT is a bidirectional Transformer encoder [1]."},
]


class _FakeLLM:
    def __init__(self, reply: str = "", error: Exception | None = None):
        self.reply, self.error, self.prompts = reply, error, []

    def generate(self, prompt, system=None, temperature=0.2, *, quick=False, info=None):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.reply


def test_first_question_is_searched_as_asked(monkeypatch):
    fake = _FakeLLM("unused")
    monkeypatch.setattr(rag, "get_llm", lambda: fake)
    assert rag.standalone_question("What is BERT?", []) == "What is BERT?"
    assert fake.prompts == []  # no extra Gemini call without history


def test_follow_up_is_rewritten_with_the_conversation(monkeypatch):
    fake = _FakeLLM('"What limitations of BERT do its authors acknowledge?"\n')
    monkeypatch.setattr(rag, "get_llm", lambda: fake)
    assert rag.standalone_question("What are its limitations?", HISTORY) == (
        "What limitations of BERT do its authors acknowledge?"
    )
    assert "What is BERT?" in fake.prompts[0]


def test_failed_or_odd_rewrite_falls_back_to_the_question(monkeypatch):
    for fake in (_FakeLLM(error=RuntimeError("503")), _FakeLLM(""), _FakeLLM("x" * 500)):
        monkeypatch.setattr(rag, "get_llm", lambda f=fake: f)
        assert rag.standalone_question("What are its limitations?", HISTORY) == "What are its limitations?"
