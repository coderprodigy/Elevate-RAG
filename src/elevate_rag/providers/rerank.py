"""Rerankers: none, lexical (offline), local cross-encoder, Cohere/Jina-style HTTP API."""

from __future__ import annotations

import math
import os
import time
from collections import Counter
from typing import Any

import httpx

from elevate_rag.cache import KVCache
from elevate_rag.config import ProviderConfig
from elevate_rag.providers.base import UsageLedger, http_error_to_exception, price
from elevate_rag.types import CallRecord
from elevate_rag.utils import ConfigError, ProviderError, Tokenizer, word_tokens


class BaseReranker:
    def __init__(
        self, cfg: ProviderConfig, *, ledger: UsageLedger, cache: KVCache | None = None
    ) -> None:
        self.cfg = cfg
        self.ledger = ledger
        self.cache = cache
        self._tok = Tokenizer()

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Relevance score per text (higher = more relevant). Cached per (query, text)."""
        if not texts:
            return []
        keys = [KVCache.key({"id": self.cfg.identity, "q": query, "t": t}) for t in texts]
        out: list[float | None] = [self.cache.get(k) if self.cache else None for k in keys]
        missing = [i for i, v in enumerate(out) if v is None]
        if missing:
            batch = [texts[i] for i in missing]
            tokens = sum(self._tok.count(t) for t in batch) + self._tok.count(query) * len(batch)
            self.ledger.check_budget(price(self.cfg, tokens, 0), self.cfg)
            t0 = time.perf_counter()
            scores = self._score(query, batch)
            self.ledger.add(
                CallRecord(
                    role="rerank",
                    provider=self.cfg.provider,
                    model=self.cfg.model,
                    prompt_tokens=tokens,
                    latency_s=time.perf_counter() - t0,
                    cost_usd=price(self.cfg, tokens, 0),
                    tokens_are_estimate=True,
                )
            )
            for j, i in enumerate(missing):
                out[i] = float(scores[j])
                if self.cache is not None:
                    self.cache.set(keys[i], float(scores[j]))
        return [float(v) for v in out]  # type: ignore[arg-type]

    def _score(self, query: str, texts: list[str]) -> list[float]:
        raise NotImplementedError


class LexicalReranker(BaseReranker):
    """BM25-style query/passage overlap in [0, 1] (offline, D-18). Not a research reranker."""

    def _score(self, query: str, texts: list[str]) -> list[float]:
        q = Counter(word_tokens(query))
        docs = [Counter(word_tokens(t)) for t in texts]
        n = len(docs)
        df = Counter(w for d in docs for w in d)
        idf = {w: math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5)) for w in q}
        denom = sum(idf.values()) or 1.0
        out = []
        for d in docs:
            dl = sum(d.values()) or 1
            s = sum(
                idf[w] * (d[w] * 2.2) / (d[w] + 1.2 * (0.25 + 0.75 * dl / 100)) for w in q if w in d
            )
            out.append(min(1.0, s / (2.2 * denom)))
        return out


class CrossEncoderReranker(BaseReranker):
    """Local cross-encoder, e.g. BAAI/bge-reranker-v2-m3 (extra `local-models`)."""

    def __init__(self, cfg: ProviderConfig, **kw: Any) -> None:
        super().__init__(cfg, **kw)
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ConfigError("install the `local-models` extra") from exc
        self._model = CrossEncoder(
            cfg.model,
            revision=cfg.revision,
            max_length=cfg.params.get("max_length", 512),
            device=cfg.params.get("device"),
        )

    def _score(self, query: str, texts: list[str]) -> list[float]:
        scores = self._model.predict(
            [(query, t) for t in texts], batch_size=self.cfg.batch_size, show_progress_bar=False
        )
        # bge-reranker outputs logits; squash to (0,1) so the R4 threshold is scale-stable.
        return [1.0 / (1.0 + math.exp(-float(s))) for s in scores]


class HttpReranker(BaseReranker):
    """Cohere v2 / Jina `/rerank` (same request/response shape)."""

    def __init__(self, cfg: ProviderConfig, **kw: Any) -> None:
        super().__init__(cfg, **kw)
        if not cfg.base_url or not cfg.api_key_env:
            raise ConfigError("http reranker needs base_url and api_key_env")
        key = os.getenv(cfg.api_key_env)
        if not key:
            raise ConfigError(f"{cfg.api_key_env} is not set")
        self._client = httpx.Client(
            base_url=cfg.base_url.rstrip("/"),
            timeout=cfg.timeout_s,
            headers={"Authorization": f"Bearer {key}"},
        )

    def _score(self, query: str, texts: list[str]) -> list[float]:
        body = {"model": self.cfg.model, "query": query, "documents": texts, "top_n": len(texts)}
        for attempt in range(self.cfg.max_retries + 1):
            try:
                r = self._client.post("/rerank", json=body)
            except httpx.TransportError as exc:
                err: Exception = exc
            else:
                if r.status_code == 200:
                    scores = [0.0] * len(texts)
                    for res in r.json()["results"]:
                        scores[res["index"]] = float(res["relevance_score"])
                    return scores
                err = http_error_to_exception(r.status_code, r.text, r.headers.get("retry-after"))
                if r.status_code not in (408, 429) and r.status_code < 500:
                    raise err
            time.sleep(min(30.0, 2.0**attempt))
        raise ProviderError(f"rerank failed after retries: {err}")


def make_reranker(
    cfg: ProviderConfig | None, *, ledger: UsageLedger, cache: KVCache | None = None
) -> BaseReranker | None:
    if cfg is None or cfg.provider == "none":
        return None
    providers: dict[str, type[BaseReranker]] = {
        "lexical": LexicalReranker,
        "cross-encoder": CrossEncoderReranker,
        "http": HttpReranker,
    }
    if cfg.provider not in providers:
        raise ConfigError(f"unknown reranker provider {cfg.provider!r}")
    return providers[cfg.provider](cfg, ledger=ledger, cache=cache)
