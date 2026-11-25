"""Cross-encoder reranking: score (query, passage) pairs jointly, keep the best ``top_n``."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Protocol

from hgrag.config import RerankerSpec


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, name: str, spec: RerankerSpec, device: str | None = None) -> None:
        from sentence_transformers import CrossEncoder

        from hgrag.data.download import hf_token
        from hgrag.index.embedders import best_device

        self.name = name
        self._spec = spec
        self._model: Any = CrossEncoder(
            spec.model,
            device=device or best_device(),
            max_length=spec.max_length,
            token=hf_token(),
        )

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        scores = self._model.predict(
            [(query, p) for p in passages],
            batch_size=self._spec.batch_size,
            show_progress_bar=False,
        )
        return [float(s) for s in scores]


class OverlapReranker:
    """Token-overlap scorer for tests (no model download)."""

    _TOKEN = re.compile(r"[a-z0-9]+")

    def __init__(self) -> None:
        self.name = "overlap"

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        query_tokens = set(self._TOKEN.findall(query.lower()))
        return [
            len(query_tokens & set(self._TOKEN.findall(p.lower()))) / (len(query_tokens) or 1)
            for p in passages
        ]
