"""Chunkers: the canonical source of chunks for every pipeline (decision D2).

* ``sentence``: pack whole sentences up to ``max_tokens``; the next chunk starts with trailing
  sentences of the previous one worth at most ``overlap_tokens``.
* ``paragraph``: one chunk per document (Wikipedia paragraphs, NQ passages); a document longer
  than ``max_tokens`` falls back to sentence chunking.

Chunk IDs are content hashes, so the vector and graph indexes agree on IDs by construction.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

from pydantic import BaseModel, ConfigDict

from hgrag.config import ChunkerSpec
from hgrag.core.hashing import stable_hash
from hgrag.data.schema import Document
from hgrag.index.tokens import count_tokens, split_by_tokens

# A sentence ends after . ! ? (plus an optional closing quote/bracket, kept) before whitespace
# and an uppercase letter, digit or opening quote. Paragraph breaks always end a sentence.
_SENTENCE_END = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"')\]]))\s+(?=[\"'(\[]?[A-Z0-9])")


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str
    doc_id: str
    title: str
    text: str
    position: int  # index of the chunk within its document
    n_tokens: int

    @property
    def indexed_text(self) -> str:
        """What retrievers see: the title gives Wikipedia paragraphs their subject."""
        return f"{self.title}\n{self.text}" if self.title else self.text


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for paragraph in text.split("\n"):
        sentences += [s.strip() for s in _SENTENCE_END.split(paragraph) if s.strip()]
    return sentences


def chunk_documents(documents: Iterable[Document], spec: ChunkerSpec) -> Iterator[Chunk]:
    for doc in documents:
        texts = (
            [doc.text]
            if spec.kind == "paragraph" and count_tokens(doc.text) <= spec.max_tokens
            else pack_sentences(split_sentences(doc.text), spec.max_tokens, spec.overlap_tokens)
        )
        for position, text in enumerate(texts):
            yield Chunk(
                chunk_id="c_" + stable_hash([doc.doc_id, position, text])[:16],
                doc_id=doc.doc_id,
                title=doc.title,
                text=text,
                position=position,
                n_tokens=count_tokens(text),
            )


def pack_sentences(sentences: list[str], max_tokens: int, overlap_tokens: int) -> list[str]:
    """Greedy sentence packing with sentence-level overlap."""
    pieces: list[tuple[str, int]] = []  # (sentence, tokens), over-long sentences pre-split
    for sentence in sentences:
        n = count_tokens(sentence)
        if n <= max_tokens:
            pieces.append((sentence, n))
        else:
            pieces += [(part, count_tokens(part)) for part in split_by_tokens(sentence, max_tokens)]

    chunks: list[str] = []
    current: list[tuple[str, int]] = []
    current_tokens = 0
    for piece in pieces:
        if current and current_tokens + piece[1] > max_tokens:
            chunks.append(" ".join(s for s, _ in current))
            current = _overlap_tail(current, overlap_tokens, room=max_tokens - piece[1])
            current_tokens = sum(n for _, n in current)
        current.append(piece)
        current_tokens += piece[1]
    if current:
        chunks.append(" ".join(s for s, _ in current))
    return chunks


def _overlap_tail(
    sentences: list[tuple[str, int]], overlap_tokens: int, room: int
) -> list[tuple[str, int]]:
    """Trailing sentences worth ≤ ``overlap_tokens`` that still leave room for the next piece."""
    tail: list[tuple[str, int]] = []
    total = 0
    for sentence, n in reversed(sentences):
        if total + n > min(overlap_tokens, room):
            break
        tail.insert(0, (sentence, n))
        total += n
    return tail
