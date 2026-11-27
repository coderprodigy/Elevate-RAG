"""Okapi BM25 over the shared chunk store, and reciprocal rank fusion (FR-V4, D-04)."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence

from elevate_rag.utils import word_tokens

Hit = tuple[str, float]

# Small English stop list; BM25's IDF already down-weights common terms, this only trims noise.
_STOP = frozenset("a an the of in on at to for from by with and or is are was were be".split())


def bm25_tokens(text: str) -> list[str]:
    return [t for t in word_tokens(text) if t not in _STOP]


class BM25Index:
    """Inverted-index BM25 (Robertson/Lucene IDF). Ties are broken by ID for determinism."""

    def __init__(self, k1: float = 0.9, b: float = 0.4) -> None:
        self.k1, self.b = k1, b
        self._ids: list[str] = []
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._len: list[int] = []
        self._avgdl = 0.0

    def build(self, ids: Sequence[str], texts: Sequence[str]) -> BM25Index:
        if len(ids) != len(texts):
            raise ValueError("ids and texts differ in length")
        for i, (doc_id, text) in enumerate(zip(ids, texts, strict=True)):
            tf = Counter(bm25_tokens(text))
            self._ids.append(doc_id)
            self._len.append(sum(tf.values()))
            for term, c in tf.items():
                self._postings[term].append((i, c))
        self._avgdl = (sum(self._len) / len(self._len)) if self._len else 0.0
        return self

    def __len__(self) -> int:
        return len(self._ids)

    def search(self, query: str, k: int) -> list[Hit]:
        n = len(self._ids)
        if n == 0:
            return []
        scores: dict[int, float] = defaultdict(float)
        for term in set(bm25_tokens(query)):
            plist = self._postings.get(term)
            if not plist:
                continue
            idf = math.log(1 + (n - len(plist) + 0.5) / (len(plist) + 0.5))
            for i, tf in plist:
                norm = tf + self.k1 * (1 - self.b + self.b * self._len[i] / (self._avgdl or 1))
                scores[i] += idf * tf * (self.k1 + 1) / norm
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], self._ids[kv[0]]))[:k]
        return [(self._ids[i], s) for i, s in ranked]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Hit]], k: int = 60, limit: int | None = None
) -> list[Hit]:
    """RRF (Cormack et al. 2009): score(d) = Σ 1 / (k + rank_i(d)), ranks starting at 1."""
    fused: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            fused[doc_id] += 1.0 / (k + rank)
    out = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    return out[:limit] if limit else out
