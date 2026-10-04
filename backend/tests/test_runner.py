from types import SimpleNamespace

from app.evaluation import runner
from app.evaluation.dataset import EvalItem
from app.generation.prompts import NOT_FOUND
from app.ingestion.chunker import chunk_id


def test_refusal_detection():
    assert runner.is_refusal(NOT_FOUND)
    assert runner.is_refusal("The sources do not contain any results on HotpotQA.")
    assert not runner.is_refusal("BERT masks 15% of tokens [1].")
    # Answering the covered half with a citation and flagging the gap is a partial answer, not a refusal
    assert not runner.is_refusal("- BERT masks 15% of tokens [4].\n- The sources do not contain information about BART.")


def test_adjacent_gold_adds_same_section_neighbours_only(monkeypatch):
    paper = "1810.04805"
    gold, before, after = chunk_id(paper, 5), chunk_id(paper, 4), chunk_id(paper, 6)
    store = {
        gold: {"paper_id": paper, "chunk_index": 5, "section": "3.1 Pre-training BERT"},
        before: {"paper_id": paper, "chunk_index": 4, "section": "3.1 Pre-training BERT"},
        after: {"paper_id": paper, "chunk_index": 6, "section": "3.2 Fine-tuning BERT"},
    }
    monkeypatch.setattr(runner.vector_store, "get_chunks",
                        lambda ids: [SimpleNamespace(id=i, payload=store[i]) for i in ids if i in store])
    item = EvalItem(id="q1", question="?", reference_answer="", gold_chunk_ids=[gold], paper_id=paper)
    assert runner.adjacent_gold([item]) == {"q1": {gold, before}}
