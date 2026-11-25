"""Text embedders. Local models run on Apple MPS (or CUDA/CPU) via sentence-transformers.

Vectors are L2-normalised, so inner product = cosine similarity everywhere.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from hgrag.config import EmbedderSpec

Vectors = npt.NDArray[np.float32]


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: Sequence[str]) -> Vectors: ...

    def embed_queries(self, texts: Sequence[str]) -> Vectors: ...


def best_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


class SentenceTransformerEmbedder:
    def __init__(self, name: str, spec: EmbedderSpec, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        from hgrag.data.download import hf_token

        self.name = name
        self._spec = spec
        self._model: Any = SentenceTransformer(
            spec.model, device=device or best_device(), token=hf_token()
        )
        self._model.max_seq_length = spec.max_seq_length

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        return self._encode(list(texts), prompt_name=None, prefix="")

    def embed_queries(self, texts: Sequence[str]) -> Vectors:
        return self._encode(
            list(texts), prompt_name=self._spec.query_prompt_name, prefix=self._spec.query_prefix
        )

    def _encode(self, texts: list[str], *, prompt_name: str | None, prefix: str) -> Vectors:
        vectors = self._model.encode(
            [prefix + t for t in texts],
            prompt_name=prompt_name,
            batch_size=self._spec.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 1000,
        )
        return np.asarray(vectors, dtype=np.float32)


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder for tests (no model download)."""

    _TOKEN = re.compile(r"[a-z0-9]+")

    def __init__(self, dim: int = 256) -> None:
        self.name = "hashing"
        self._dim = dim

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return np.stack([self._vector(t) for t in texts]).astype(np.float32)

    def embed_queries(self, texts: Sequence[str]) -> Vectors:
        return self.embed_documents(texts)

    def _vector(self, text: str) -> Vectors:
        vector = np.zeros(self._dim, dtype=np.float32)
        for token in self._TOKEN.findall(text.lower()):
            vector[int(hashlib.md5(token.encode()).hexdigest(), 16) % self._dim] += 1.0
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector
