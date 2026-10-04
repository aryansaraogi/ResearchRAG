"""Paper summaries, the citation graph and paper comparison (LLM and vector search are faked)."""

from app.generation import compare as cmp
from app.generation.summaries import Passage, select_passages, summarize
from app.ingestion import references as refs
from app.models import Paper
from app.retrieval.hybrid import RetrievedChunk

ATTENTION = Paper(id="1706.03762", title="Attention Is All You Need", source="arxiv")
BERT = Paper(id="1810.04805", title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
             source="arxiv")
SHORT = Paper(id="x1", title="Deep Learning", source="upload")


# ---------- citation graph ----------

def test_title_match_survives_line_break_hyphens_and_punctuation():
    reference = "Jacob Devlin et al. 2019. BERT: Pre-\ntraining of deep bidirectional transformers for language understanding. NAACL."
    assert refs.cites(reference, BERT)


def test_arxiv_id_match_and_no_false_prefix_match():
    assert refs.cites("Vaswani et al. arXiv preprint arXiv:1706.03762, 2017.", ATTENTION)
    assert not refs.cites("Some paper. arXiv:1706.037621", ATTENTION)


def test_short_generic_titles_are_not_matched():
    assert not refs.cites("Goodfellow et al. Deep learning. MIT Press, 2016.", SHORT)


def test_citation_graph_edges(monkeypatch):
    lists = {"1810.04805": "Ashish Vaswani et al. 2017. Attention is all you need. In NIPS.", "1706.03762": ""}
    monkeypatch.setattr(refs, "load_references", lambda p: lists[p.id])
    graph = refs.citation_graph([ATTENTION, BERT])
    assert graph["edges"] == [{"source": "1810.04805", "target": "1706.03762"}]
    assert {n["id"]: n["n_references"] for n in graph["nodes"]} == {"1706.03762": 0, "1810.04805": 1}


def test_reference_counting_handles_wrapped_entries():
    numbered = "[1] Ba et al. Layer normalization.\narXiv:1607.06450, 2016.\n[2] Bahdanau et al. Neural machine\ntranslation. 2014.\n[3] Britz et al. 2017."
    author_year = "Alan Akbik and Roland Vollgraf.\n2018. Contextual string embeddings.\nRie Ando and Tong Zhang. 2005. A framework."
    assert refs.count_references(numbered) == 3
    assert refs.count_references(author_year) == 2


# ---------- summaries ----------

def _p(section, kind, words=100):
    return Passage(section=section, section_type=kind, page=1, text=" ".join(["w"] * words))


def test_summary_reads_abstract_and_conclusion_before_introduction_within_budget():
    passages = [_p("Abstract", "abstract"), _p("1 Introduction", "introduction", 3000),
                _p("3 Method", "method"), _p("6 Conclusion", "conclusion"), _p("5.4 Limitations", "other")]
    chosen = select_passages(passages, max_words=500)
    assert [p.section for p in chosen] == ["Abstract", "6 Conclusion", "5.4 Limitations"]  # reading order kept


class _FakeLLM:
    def __init__(self, reply):
        self.reply = reply

    def generate_json(self, prompt, system=None, schema=None, temperature=0.0, info=None, *, quick=False):
        if info is not None:
            info["model"] = "fake-model"
        return self.reply


def test_summary_drops_invented_section_labels():
    reply = {"tldr": "A paper.", "contributions": [{"text": "Does X.", "section": "Abstract"},
                                                    {"text": "Does Y.", "section": "7 Appendix Z"}],
             "limitations": []}
    out = summarize(_FakeLLM(reply), "T", [_p("Abstract", "abstract")])
    assert [c["section"] for c in out["contributions"]] == ["Abstract", ""] and out["model"] == "fake-model"


# ---------- compare ----------

def _chunk(cid, page):
    return RetrievedChunk(id=cid, paper_id="1706.03762", title="Attention Is All You Need", authors=[], year=2017,
                          section="3 Model", section_type="method", page_start=page, page_end=page, text="...", score=1.0)


def test_compare_cells_get_citations_and_missing_aspects_say_so(monkeypatch):
    found = {"Method": [_chunk("c1", 3)], "Results": [_chunk("c2", 8), _chunk("c1", 3)]}
    monkeypatch.setattr(cmp, "search", lambda q, **kw: found.get(q.split(":")[0], []))
    reply = {"cells": [{"aspect": "Method", "text": "Self-attention only [1]."},
                       {"aspect": "Results", "text": "28.4 BLEU on EN-DE [2][9]."}]}
    monkeypatch.setattr(cmp, "get_llm", lambda: _FakeLLM(reply))
    col = cmp.compare_paper("1706.03762", "Attention Is All You Need", 2017,
                            {"Method": "method", "Results": "results", "Limitations": "limits"})
    assert [s.id for s in col.sources] == ["c1", "c2"]  # de-duplicated across aspects
    assert col.cells["Method"].citations[0].page_start == 3
    assert col.cells["Results"].text == "28.4 BLEU on EN-DE [2]."  # [9] doesn't exist, so it is stripped
    assert col.cells["Limitations"].text == cmp.NOT_COVERED
