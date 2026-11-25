"""Prompt templates. Changing any of these changes results, so each has a version string that
is recorded in every run config."""

from __future__ import annotations

from hgrag.generation.answers import REFUSAL

CLOSED_BOOK_VERSION = "closed_book/v1"

CLOSED_BOOK_SYSTEM = f"""You answer questions from your own knowledge.
- For a factual question, reply with only the answer: a name, number, date or short phrase.
- For a question that asks for an explanation, summary or comparison, reply in at most 5 sentences.
- If you do not know the answer, reply exactly: {REFUSAL}
Do not explain your reasoning."""


def closed_book_prompt(question: str) -> str:
    return f"Question: {question}"


RAG_VERSION = "rag/v1"

RAG_SYSTEM = f"""You answer questions using only the numbered passages provided.
- Cite the passages you used with their numbers, like [1] or [2][3].
- For a factual question, reply with only the answer (a name, number, date or short phrase) \
followed by its citations.
- For a question that asks for an explanation, summary or comparison, reply in at most 5 \
sentences, citing as you go.
- If the passages do not contain the answer, reply exactly: {REFUSAL}
Do not use outside knowledge and do not explain your reasoning."""


def rag_prompt(question: str, passages: list[tuple[str, str]]) -> str:
    """``passages`` = (title, text) in rank order; numbered from 1 for citations."""
    numbered = "\n\n".join(
        f"[{i}] ({title})\n{text}" if title else f"[{i}]\n{text}"
        for i, (title, text) in enumerate(passages, 1)
    )
    return f"Passages:\n{numbered}\n\nQuestion: {question}"
