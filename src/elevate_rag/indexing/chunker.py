"""Deterministic, sentence-aware token chunking (FR-V1).

* `sentence`: greedily pack whole sentences up to `chunk_tokens`; the next chunk starts with the
  trailing sentences of the previous one totalling at most `overlap_tokens`. A sentence longer
  than a chunk is split into token windows (the only place a sentence is cut).
* `paragraph`: one chunk per paragraph (blank-line separated); over-long paragraphs fall back to
  sentence packing. Used for Wikipedia datasets whose gold evidence is a paragraph.

Chunk IDs are `<doc_id>::c<index>` — stable for a given document and chunking config.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from elevate_rag.config import ChunkingConfig
from elevate_rag.types import Chunk, Document
from elevate_rag.utils import Tokenizer

# Sentence end: . ! ? (optionally followed by quotes/brackets) then whitespace; or a newline run.
_SENT_BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]*\s+|\n\s*\n|\n(?=[-*•\d])")
_PARA_BOUNDARY = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    n_tokens: int


def chunk_id(doc_id: str, index: int) -> str:
    return f"{doc_id}::c{index:04d}"


class Chunker:
    def __init__(self, cfg: ChunkingConfig) -> None:
        self.cfg = cfg
        self.tok = Tokenizer(cfg.encoding)

    # -- public --------------------------------------------------------------------------------
    def chunk_documents(self, docs: Iterable[Document]) -> Iterator[Chunk]:
        for d in docs:
            yield from self.chunk_document(d)

    def chunk_document(self, doc: Document) -> list[Chunk]:
        text = doc.text
        if self.cfg.strategy == "paragraph":
            spans: list[_Span] = []
            for ps, pe in self._split(text, 0, len(text), _PARA_BOUNDARY):
                n = self.tok.count(text[ps:pe])
                spans += (
                    [_Span(ps, pe, n)]
                    if n <= self._body_budget(doc)
                    else self._pack(text, self._sentences(text, ps, pe), self._body_budget(doc))
                )
        else:
            spans = self._pack(text, self._sentences(text, 0, len(text)), self._body_budget(doc))
        prefix = f"{doc.title}\n" if self.cfg.prepend_title and doc.title else ""
        out: list[Chunk] = []
        for s in spans:
            body = text[s.start : s.end].strip()
            if not body:
                continue
            full = prefix + body
            out.append(
                Chunk(
                    chunk_id=chunk_id(doc.doc_id, len(out)),
                    doc_id=doc.doc_id,
                    index=len(out),
                    text=full,
                    n_tokens=self.tok.count(full),
                    start_char=s.start,
                    end_char=s.end,
                    title=doc.title,
                )
            )
        return out

    # -- internals -----------------------------------------------------------------------------
    def _body_budget(self, doc: Document) -> int:
        title_tokens = self.tok.count(doc.title) + 1 if self.cfg.prepend_title and doc.title else 0
        return max(16, self.cfg.chunk_tokens - title_tokens)

    @staticmethod
    def _split(text: str, start: int, end: int, pattern: re.Pattern[str]) -> list[tuple[int, int]]:
        out = []
        pos = start
        for m in pattern.finditer(text, start, end):
            if m.start() > pos and text[pos : m.start()].strip():
                out.append((pos, m.start()))
            pos = m.end()
        if pos < end and text[pos:end].strip():
            out.append((pos, end))
        return out

    def _sentences(self, text: str, start: int, end: int) -> list[_Span]:
        return [
            _Span(s, e, self.tok.count(text[s:e]))
            for s, e in self._split(text, start, end, _SENT_BOUNDARY)
        ]

    def _hard_split(self, text: str, span: _Span, budget: int) -> list[_Span]:
        """Split one over-long sentence into token windows, mapping back to char offsets."""
        toks = self.tok.encode(text[span.start : span.end])
        out = []
        pos = span.start
        for i in range(0, len(toks), budget):
            piece = self.tok.decode(toks[i : i + budget])
            found = text.find(piece.strip()[:20], pos) if piece.strip() else -1
            s = found if found >= 0 else pos
            e = min(span.end, s + len(piece.strip())) if i + budget < len(toks) else span.end
            out.append(_Span(s, max(e, s + 1), min(budget, len(toks) - i)))
            pos = e
        return out

    def _pack(self, text: str, sents: list[_Span], budget: int) -> list[_Span]:
        units: list[_Span] = []
        for s in sents:
            units += self._hard_split(text, s, budget) if s.n_tokens > budget else [s]
        chunks: list[_Span] = []
        cur: list[_Span] = []
        cur_tokens = 0
        for u in units:
            # exact count of the joined span (sum of sentence counts can under-count by ~1/join)
            if cur and self.tok.count(text[cur[0].start : u.end]) > budget:
                chunks.append(_Span(cur[0].start, cur[-1].end, cur_tokens))
                overlap: list[_Span] = []
                ov_tokens = 0
                for prev in reversed(cur):
                    if ov_tokens + prev.n_tokens > self.cfg.overlap_tokens:
                        break
                    overlap.insert(0, prev)
                    ov_tokens += prev.n_tokens
                # never let the overlap alone fill the chunk
                if ov_tokens + u.n_tokens > budget:
                    overlap, ov_tokens = [], 0
                cur, cur_tokens = overlap, ov_tokens
            cur.append(u)
            cur_tokens = self.tok.count(text[cur[0].start : cur[-1].end])
        if cur:
            chunks.append(_Span(cur[0].start, cur[-1].end, cur_tokens))
        return chunks
