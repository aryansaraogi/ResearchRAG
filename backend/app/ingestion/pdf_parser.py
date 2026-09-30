"""Parse research-paper PDFs into sections of paragraphs using PyMuPDF.

Headings are detected from typography (bold / larger than body text) combined
with patterns common in papers ("3.1 Method", "IV. RESULTS", "Abstract").
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

KNOWN_HEADINGS = {
    "abstract", "introduction", "background", "related work", "related works",
    "preliminaries", "method", "methods", "methodology", "approach", "model",
    "experiments", "experimental setup", "experimental results", "evaluation",
    "results", "discussion", "analysis", "limitations", "conclusion",
    "conclusions", "future work", "conclusion and future work",
    "acknowledgements", "acknowledgments", "acknowledgement", "references",
    "bibliography", "appendix", "appendices", "broader impact",
}
NUMBERED_HEADING = re.compile(
    r"^(?:(?:\d+(?:\.\d+){0,3})\.?|[IVX]{1,5}\.|[A-H](?:\.\d+){0,2}\.?)\s+[A-Z][\w\-–:,'&()/ ]{1,90}$"
)
NUMBER_TOKEN = re.compile(r"^(?:\d{1,2}(?:\.\d{1,2}){0,3}|[IVX]{1,5}|[A-H](?:\.\d{1,2}){0,2})\.?$")
REFERENCE_SECTIONS = {"references", "bibliography"}


@dataclass
class Paragraph:
    text: str
    page: int  # 1-based


@dataclass
class Section:
    title: str
    paragraphs: list[Paragraph] = field(default_factory=list)


@dataclass
class ParsedDocument:
    title: str
    num_pages: int
    sections: list[Section]


@dataclass
class _Line:
    text: str
    size: float
    bold: bool
    page: int


def _normalize_heading(text: str) -> str:
    stripped = re.sub(r"^(?:\d+(?:\.\d+)*\.?|[IVX]{1,5}\.|[A-H](?:\.\d+)*\.?)\s+", "", text)
    return stripped.strip().rstrip(".:").lower()


def _is_number_token(line: _Line) -> bool:
    """A bold section number on its own span, e.g. '3' or '3.1' before 'Model Architecture'."""
    return line.bold and bool(NUMBER_TOKEN.match(line.text.strip()))


def _is_heading(line: _Line, body_size: float, numbered: bool = False) -> bool:
    """numbered=True when the line directly follows a detached bold section number."""
    text = line.text.strip()
    if not text or len(text) > 100 or len(text.split()) > 12 or not text[0].isupper():
        return False
    if text.endswith((".", ",", ";")) and _normalize_heading(text) not in KNOWN_HEADINGS:
        return False
    emphasized = line.bold or line.size >= body_size + 0.8
    # Table headers and captions are usually set smaller than body text
    at_least_body = line.size >= body_size - 0.1
    if numbered:
        return emphasized and at_least_body
    if NUMBERED_HEADING.match(text):
        return emphasized
    if _normalize_heading(text) in KNOWN_HEADINGS:
        return at_least_body and (emphasized or text.isupper())
    return False


def _join_lines(lines: list[str]) -> str:
    out = ""
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if out.endswith("-") and ln[:1].islower():
            out = out[:-1] + ln  # de-hyphenate words split across lines
        elif out:
            out += " " + ln
        else:
            out = ln
    return re.sub(r"\s+", " ", out).strip()


def _reading_order(blocks: list[dict], page_width: float) -> list[dict]:
    """Order text blocks for one- or two-column layouts.

    Full-width blocks (titles, wide figures/tables) act as separators; between them the
    left column is read top-to-bottom before the right column.
    """
    mid = page_width / 2
    ordered: list[dict] = []
    left: list[dict] = []
    right: list[dict] = []

    def flush_columns() -> None:
        ordered.extend(sorted(left, key=lambda b: b["bbox"][1]))
        ordered.extend(sorted(right, key=lambda b: b["bbox"][1]))
        left.clear()
        right.clear()

    for b in sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0])):
        x0, _, x1, _ = b["bbox"]
        if x1 <= mid + 10:
            left.append(b)
        elif x0 >= mid - 10:
            right.append(b)
        else:  # spans both columns
            flush_columns()
            ordered.append(b)
    flush_columns()
    return ordered


def _clean(text: str) -> str:
    # NFKC folds ligatures (ﬁ -> fi) and full-width forms so keyword search matches
    return unicodedata.normalize("NFKC", text)


def _extract_lines(doc: pymupdf.Document) -> list[list[list[_Line]]]:
    """Return pages -> blocks -> lines, skipping non-horizontal text (e.g. arXiv sidebar)."""
    pages = []
    for page_no, page in enumerate(doc, start=1):
        blocks = []
        text_blocks = [b for b in page.get_text("dict")["blocks"] if b.get("type") == 0]
        for block in _reading_order(text_blocks, page.rect.width):
            lines = []
            for line in block["lines"]:
                dx, dy = line.get("dir", (1, 0))
                if abs(dy) > 0.1:
                    continue
                spans = [s for s in line["spans"] if s["text"].strip()]
                if not spans:
                    continue
                text = _clean("".join(s["text"] for s in line["spans"]))
                size = max(s["size"] for s in spans)
                bold = all((s["flags"] & 16) or "bold" in s["font"].lower() for s in spans)
                lines.append(_Line(text=text, size=round(size, 1), bold=bold, page=page_no))
            if lines:
                blocks.append(lines)
        pages.append(blocks)
    return pages


def _body_font_size(pages: list[list[list[_Line]]]) -> float:
    counter: Counter[float] = Counter()
    for blocks in pages:
        for lines in blocks:
            for ln in lines:
                counter[ln.size] += len(ln.text)
    return counter.most_common(1)[0][0] if counter else 10.0


def _repeated_lines(pages: list[list[list[_Line]]]) -> set[str]:
    """Running headers/footers: short lines that appear on many pages."""
    if len(pages) < 3:
        return set()
    counter: Counter[str] = Counter()
    for blocks in pages:
        # Ignore very short tokens (section numbers, table digits); page numbers are handled separately
        seen = {ln.text.strip() for lines in blocks for ln in lines if 3 < len(ln.text.strip()) < 80}
        counter.update(seen)
    return {t for t, c in counter.items() if c >= max(3, len(pages) // 2)}


def _guess_title(pages: list[list[list[_Line]]], body_size: float) -> str:
    if not pages or not pages[0]:
        return ""
    first = [ln for lines in pages[0] for ln in lines]
    max_size = max(ln.size for ln in first)
    if max_size <= body_size:
        return ""
    title_lines = []
    for ln in first:
        if ln.size == max_size:
            title_lines.append(ln.text)
        elif title_lines:
            break
    return _join_lines(title_lines)[:300]


def parse_pdf(path: str | Path) -> ParsedDocument:
    with pymupdf.open(path) as doc:
        pages = _extract_lines(doc)
        num_pages = doc.page_count

    body_size = _body_font_size(pages)
    repeated = _repeated_lines(pages)
    title = _guess_title(pages, body_size)

    sections: list[Section] = [Section(title="Front Matter")]
    pending_number: str | None = None  # detached section number waiting for its heading text
    for blocks in pages:
        for lines in blocks:
            buffer: list[str] = []
            buffer_page = lines[0].page

            def flush() -> None:
                text = _join_lines(buffer)
                if len(text) > 1:
                    sections[-1].paragraphs.append(Paragraph(text=text, page=buffer_page))
                buffer.clear()

            for ln in lines:
                stripped = ln.text.strip()
                if stripped in repeated:
                    continue
                if _is_number_token(ln) and ln.size >= body_size - 0.1:
                    pending_number = stripped.rstrip(".")
                    continue
                if re.fullmatch(r"\d{1,3}", stripped):  # page numbers
                    continue
                if _is_heading(ln, body_size, numbered=pending_number is not None):
                    flush()
                    heading = _join_lines([stripped])
                    sections.append(Section(title=f"{pending_number} {heading}" if pending_number else heading))
                    buffer_page = ln.page
                    pending_number = None
                    continue
                pending_number = None
                if not buffer:
                    buffer_page = ln.page
                buffer.append(ln.text)
            flush()

    sections = [s for s in sections if s.paragraphs]
    return ParsedDocument(title=title, num_pages=num_pages, sections=sections)


def is_reference_section(title: str) -> bool:
    return _normalize_heading(title) in REFERENCE_SECTIONS


SECTION_TYPES = {
    "abstract": ("abstract",),
    "introduction": ("introduction", "overview"),
    "related_work": ("related work", "background", "prior work", "literature", "preliminar"),
    "method": ("method", "approach", "model", "architecture", "framework", "algorithm", "proposed"),
    "experiments": ("experiment", "setup", "evaluation", "training", "dataset", "implementation"),
    "results": ("result", "analysis", "ablation", "comparison", "performance"),
    "discussion": ("discussion", "limitation", "broader impact", "ethic"),
    "conclusion": ("conclusion", "future work", "summary"),
    "appendix": ("appendix", "appendices", "supplementary"),
}


def section_type(title: str) -> str:
    """Map a free-form section title to a canonical type used for metadata filtering."""
    norm = _normalize_heading(title)
    if norm == "front matter":
        return "front_matter"
    for kind, keys in SECTION_TYPES.items():
        if any(k in norm for k in keys):
            return kind
    if re.match(r"^[A-H](?:\.\d+)*\.?\s", title.strip()):
        return "appendix"
    return "other"
