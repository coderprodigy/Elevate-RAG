"""BM25 lexical index (``bm25s``, English stopwords + Snowball stemming)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import bm25s
import Stemmer


class BM25Index:
    def __init__(self, retriever: Any, ids: list[str]) -> None:
        self._retriever = retriever
        self._ids = ids
        self._stemmer = Stemmer.Stemmer("english")

    @classmethod
    def build(cls, texts: list[str], ids: list[str]) -> BM25Index:
        stemmer = Stemmer.Stemmer("english")
        tokens = bm25s.tokenize(texts, stopwords="en", stemmer=stemmer, show_progress=False)
        retriever = bm25s.BM25()
        retriever.index(tokens, show_progress=False)
        return cls(retriever, ids)

    def save(self, path: Path) -> None:
        self._retriever.save(str(path))

    @classmethod
    def load(cls, path: Path, ids: list[str]) -> BM25Index:
        return cls(bm25s.BM25.load(str(path)), ids)

    def search(self, queries: list[str], k: int) -> list[list[tuple[str, float]]]:
        """Top ``k`` (chunk_id, score) per query, best first; zero-score hits dropped."""
        tokens = bm25s.tokenize(queries, stopwords="en", stemmer=self._stemmer, show_progress=False)
        k = min(k, len(self._ids))
        rows, scores = self._retriever.retrieve(tokens, k=k, show_progress=False)
        return [
            [(self._ids[int(r)], float(s)) for r, s in zip(row, score, strict=True) if s > 0]
            for row, score in zip(rows, scores, strict=True)
        ]
