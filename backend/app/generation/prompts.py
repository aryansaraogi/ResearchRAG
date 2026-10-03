"""Prompts for citation-grounded answering."""

from __future__ import annotations

from app.retrieval.hybrid import RetrievedChunk

NOT_FOUND = "I couldn't find this in the indexed papers."

ANSWER_SYSTEM = f"""You are ResearchRAG, an assistant that answers questions about research papers.

Rules:
- Answer ONLY from the numbered sources provided. Do not use outside knowledge.
- Cite every factual sentence with the source number(s) in square brackets, e.g. [1] or [2][4].
  Place citations at the end of the sentence they support. Cite only sources that actually support the sentence.
- If the sources do not contain the answer, reply exactly: "{NOT_FOUND}"
  If they contain only part of it, answer that part and state what is missing.
- When sources from different papers disagree or differ, say so and cite each.
- Be concise and technical. Use short paragraphs or bullet points. Use Markdown; LaTeX math inline as $...$.
"""


CONDENSE_SYSTEM = "You turn follow-up questions about research papers into standalone search queries."


def condense_prompt(question: str, history: list[dict]) -> str:
    # Long answers add little for resolving "it"/"they"; keep the tail of the conversation, trimmed
    turns = [f"{m['role'].upper()}: {m['content'][:600]}" for m in history[-4:]]
    return (
        "Conversation so far:\n" + "\n".join(turns) + f"\n\nFollow-up question: {question}\n\n"
        "Rewrite the follow-up as ONE self-contained question that can be searched without the conversation: "
        'replace words like "it", "they", "that model" or "the second one" with what they refer to, and keep '
        "technical terms exact. If it is already self-contained, return it unchanged. Reply with the question only."
    )


def format_sources(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        pages = f"p. {c.page_start}" if c.page_start == c.page_end else f"pp. {c.page_start}-{c.page_end}"
        year = f", {c.year}" if c.year else ""
        blocks.append(f"[{i}] {c.title}{year} — §{c.section}, {pages}\n{c.text}")
    return "\n\n".join(blocks)


def answer_prompt(question: str, chunks: list[RetrievedChunk], history: list[dict] | None = None) -> str:
    convo = ""
    if history:
        turns = [f"{m['role'].upper()}: {m['content']}" for m in history[-6:]]
        convo = "Previous conversation (for resolving follow-up references only):\n" + "\n".join(turns) + "\n\n"
    return f"{convo}Sources:\n\n{format_sources(chunks)}\n\nQuestion: {question}\n\nAnswer with citations:"
