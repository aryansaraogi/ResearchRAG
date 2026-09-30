from app.ingestion.chunker import chunk_document
from app.ingestion.pdf_parser import Paragraph, ParsedDocument, Section, section_type


def _doc():
    long_para = " ".join(f"Sentence number {i} explains a detail of the method." for i in range(80))
    return ParsedDocument(title="T", num_pages=3, sections=[
        Section("Abstract", [Paragraph("We propose a new model for things. " * 5, 1)]),
        Section("3 Method", [Paragraph(long_para, 2), Paragraph("Follow-up paragraph on page three. " * 5, 3)]),
        Section("References", [Paragraph("[1] Someone. A paper. 2020. " * 10, 3)]),
    ])


def test_chunks_respect_sections_and_skip_references():
    chunks = chunk_document(_doc(), "p1", chunk_tokens=200, overlap=0.15)
    assert {c.section for c in chunks} == {"Abstract", "3 Method"}
    assert all(len(c.text.split()) <= 150 for c in chunks)
    method = [c for c in chunks if c.section == "3 Method"]
    assert len(method) >= 3
    assert method[-1].page_end == 3
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert len({c.id for c in chunks}) == len(chunks)


def test_overlap_between_consecutive_chunks():
    chunks = [c for c in chunk_document(_doc(), "p1", chunk_tokens=200, overlap=0.15) if c.section == "3 Method"]
    tail = chunks[0].text.split()[-10:]
    assert " ".join(tail) in chunks[1].text


def test_section_type_mapping():
    assert section_type("3.1 Scaled Dot-Product Attention") == "other"
    assert section_type("2 Related Work") == "related_work"
    assert section_type("5 Experiments") == "experiments"
    assert section_type("6 Conclusion") == "conclusion"
    assert section_type("A Hyperparameters") == "appendix"
