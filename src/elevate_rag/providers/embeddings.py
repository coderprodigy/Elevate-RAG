"""Embedding providers with per-text caching and dimension validation (FR-V2, FR-M5).

All embedders return L2-normalised float32 matrices, so inner product == cosine similarity
everywhere (exact store, FAISS, Qdrant, ...). `kind` distinguishes query vs document encoding for
asymmetric models; LightRAG passes the same distinction via `context=` (D-04).
"""

from __future__ import annotations

import hashlib
import itertools
import os
import time
from typing import Any, Literal

import httpx
import numpy as np

from elevate_rag.cache import KVCache
from elevate_rag.config import ProviderConfig, require_resolved
from elevate_rag.providers.base import UsageLedger, http_error_to_exception, price
from elevate_rag.types import CallRecord
from elevate_rag.utils import ConfigError, ProviderError, Tokenizer, word_tokens

Kind = Literal["query", "document"]


def l2_normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class BaseEmbedder:
    def __init__(
        self, cfg: ProviderConfig, *, ledger: UsageLedger, cache: KVCache | None = None
    ) -> None:
        self.cfg = cfg
        self.ledger = ledger
        self.cache = cache
        self._tok = Tokenizer()
        self._dim: int | None = cfg.dimension

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._dim = int(self.embed(["dimension probe"], kind="document").shape[1])
        return self._dim

    @property
    def model_id(self) -> str:
        return f"{self.cfg.provider}:{self.cfg.model}@{self.cfg.revision or 'latest'}"

    def embed(self, texts: list[str], kind: Kind = "document") -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim or 0), dtype=np.float32)
        keys = [KVCache.key({"id": self.cfg.identity, "kind": kind, "text": t}) for t in texts]
        out: list[np.ndarray | None] = [None] * len(texts)
        if self.cache is not None:
            for i, k in enumerate(keys):
                if (hit := self.cache.get(k)) is not None:
                    out[i] = np.asarray(hit, dtype=np.float32)
        missing = [i for i, v in enumerate(out) if v is None]
        cached_n = len(texts) - len(missing)
        for start in range(0, len(missing), self.cfg.batch_size):
            idx = missing[start : start + self.cfg.batch_size]
            batch = [texts[i] for i in idx]
            tokens = sum(self._tok.count(t) for t in batch)
            self.ledger.check_budget(price(self.cfg, tokens, 0), self.cfg)
            t0 = time.perf_counter()
            vecs = l2_normalize(self._embed_batch(batch, kind))
            self.ledger.add(
                CallRecord(
                    role=f"embed:{kind}",
                    provider=self.cfg.provider,
                    model=self.cfg.model,
                    prompt_tokens=tokens,
                    latency_s=time.perf_counter() - t0,
                    cost_usd=price(self.cfg, tokens, 0),
                    tokens_are_estimate=True,
                )
            )
            for j, i in enumerate(idx):
                out[i] = vecs[j]
                if self.cache is not None:
                    self.cache.set(keys[i], vecs[j].round(7).tolist())
        if cached_n:
            self.ledger.add(
                CallRecord(
                    role=f"embed:{kind}",
                    provider=self.cfg.provider,
                    model=self.cfg.model,
                    cached=True,
                    cost_usd=0.0,
                    extra={"texts": cached_n},
                )
            )
        mat = np.vstack([v for v in out if v is not None]).astype(np.float32)
        if self._dim is not None and mat.shape[1] != self._dim:
            raise ProviderError(
                f"{self.model_id}: expected dimension {self._dim}, got {mat.shape[1]} (FR-M5)"
            )
        self._dim = mat.shape[1]
        return mat

    def _embed_batch(self, texts: list[str], kind: Kind) -> np.ndarray:
        raise NotImplementedError


class HashEmbedder(BaseEmbedder):
    """Signed feature hashing of word unigrams + bigrams, sublinear TF (D-18). Offline only."""

    def __init__(self, cfg: ProviderConfig, **kw: Any) -> None:
        super().__init__(cfg, **kw)
        self._dim = cfg.dimension or 384

    def _embed_batch(self, texts: list[str], kind: Kind) -> np.ndarray:
        assert self._dim is not None
        m = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            toks = word_tokens(text)
            feats = toks + [f"{a}_{b}" for a, b in itertools.pairwise(toks)]
            counts: dict[int, float] = {}
            for f in feats:
                h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
                idx, sign = h % self._dim, (1.0 if (h >> 63) & 1 else -1.0)
                counts[idx] = counts.get(idx, 0.0) + sign
            for idx, c in counts.items():
                m[row, idx] = np.sign(c) * (1 + np.log(abs(c))) if c else 0.0
        return m


class SentenceTransformerEmbedder(BaseEmbedder):
    """Local open-weight embedders (bge-small, MiniLM, Qwen3-Embedding) — extra `local-models`."""

    def __init__(self, cfg: ProviderConfig, **kw: Any) -> None:
        super().__init__(cfg, **kw)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ConfigError(
                "install the `local-models` extra: uv sync --extra local-models"
            ) from exc
        self._model = SentenceTransformer(
            cfg.model, revision=cfg.revision, device=cfg.params.get("device")
        )
        self._query_prompt = cfg.params.get("query_prompt_name")

    def _embed_batch(self, texts: list[str], kind: Kind) -> np.ndarray:
        kw: dict[str, Any] = {
            "batch_size": self.cfg.batch_size,
            "normalize_embeddings": True,
            "convert_to_numpy": True,
            "show_progress_bar": False,
        }
        if kind == "query" and self._query_prompt:
            kw["prompt_name"] = self._query_prompt
        return np.asarray(self._model.encode(texts, **kw), dtype=np.float32)


class OpenAICompatEmbedder(BaseEmbedder):
    """`/v1/embeddings` (OpenAI, Gemini OpenAI-compat endpoint, vLLM, Ollama, TEI)."""

    def __init__(self, cfg: ProviderConfig, **kw: Any) -> None:
        super().__init__(cfg, **kw)
        base = require_resolved(cfg.base_url, cfg.model) or os.getenv("OPENAI_COMPAT_BASE_URL")
        if not base:
            raise ConfigError(f"{cfg.model}: base_url is required for openai-compatible")
        key = os.getenv(cfg.api_key_env) if cfg.api_key_env else None
        self._client = httpx.Client(
            base_url=base.rstrip("/"),
            timeout=cfg.timeout_s,
            headers={"Authorization": f"Bearer {key}"} if key else {},
        )

    def _embed_batch(self, texts: list[str], kind: Kind) -> np.ndarray:
        prefix = self.cfg.params.get(f"{kind}_prefix", "")
        body = {"model": self.cfg.model, "input": [prefix + t for t in texts]}
        if self.cfg.params.get("dimensions"):
            body["dimensions"] = self.cfg.params["dimensions"]
        last: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                r = self._client.post("/embeddings", json=body)
            except httpx.TransportError as exc:
                last = exc
            else:
                if r.status_code == 200:
                    data = sorted(r.json()["data"], key=lambda d: d["index"])
                    return np.asarray([d["embedding"] for d in data], dtype=np.float32)
                last = http_error_to_exception(r.status_code, r.text, r.headers.get("retry-after"))
                if r.status_code not in (408, 429) and r.status_code < 500:
                    raise last
            time.sleep(min(30.0, 2.0**attempt))
        raise ProviderError(f"embedding request failed after retries: {last}")


def make_embedder(
    cfg: ProviderConfig, *, ledger: UsageLedger, cache: KVCache | None = None
) -> BaseEmbedder:
    providers: dict[str, type[BaseEmbedder]] = {
        "hash": HashEmbedder,
        "sentence-transformers": SentenceTransformerEmbedder,
        "openai-compatible": OpenAICompatEmbedder,
    }
    if cfg.provider not in providers:
        raise ConfigError(f"unknown embedder provider {cfg.provider!r}")
    return providers[cfg.provider](cfg, ledger=ledger, cache=cache)
