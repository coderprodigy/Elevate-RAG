"""LLM providers: Anthropic (official SDK), OpenAI-compatible HTTP, and a deterministic offline
provider used for tests and the offline demo (D-18).

Every call goes through :class:`BaseLLM.complete`, which adds caching, bounded concurrency,
retries, budget checks and a :class:`CallRecord` in the shared ledger.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import httpx

from elevate_rag.cache import KVCache
from elevate_rag.config import ProviderConfig, require_resolved
from elevate_rag.providers.base import (
    LoopLocalSemaphore,
    RetryableError,
    UsageLedger,
    http_error_to_exception,
    price,
    retry_async,
)
from elevate_rag.types import CallRecord
from elevate_rag.utils import ConfigError, ProviderError, Tokenizer, word_tokens

log = logging.getLogger(__name__)

Message = dict[str, str]  # {"role": "user"|"assistant", "content": str}


@dataclass
class LLMResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cached: bool = False
    stop_reason: str | None = None
    tokens_are_estimate: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class BaseLLM:
    """Common wrapper. Subclasses implement :meth:`_call`."""

    supports_sampling_controls = True

    def __init__(
        self, cfg: ProviderConfig, *, role: str, ledger: UsageLedger, cache: KVCache | None = None
    ) -> None:
        self.cfg = cfg
        self.role = role
        self.ledger = ledger
        self.cache = cache
        self._sem = LoopLocalSemaphore(cfg.max_concurrency)
        self._tok = Tokenizer()

    # -- public ------------------------------------------------------------------------------
    async def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        task: str | None = None,
        hints: dict[str, Any] | None = None,
    ) -> LLMResult:
        """`task` labels the call in the ledger; `hints` are structured inputs used only by the
        offline provider (real providers read the prompt)."""
        role = f"{self.role}:{task}" if task else self.role
        key_payload = {
            "id": self.cfg.identity,
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        key = KVCache.key(key_payload)
        if self.cache is not None and (hit := self.cache.get(key)) is not None:
            res = LLMResult(**hit)
            res.cached = True
            self.ledger.add(
                CallRecord(
                    role=role,
                    provider=self.cfg.provider,
                    model=res.model,
                    prompt_tokens=res.prompt_tokens,
                    completion_tokens=res.completion_tokens,
                    cached=True,
                    cost_usd=0.0,
                    tokens_are_estimate=res.tokens_are_estimate,
                )
            )
            return res

        est_prompt = sum(self._tok.count(m["content"]) for m in messages) + (
            self._tok.count(system) if system else 0
        )
        self.ledger.check_budget(price(self.cfg, est_prompt, max_tokens), self.cfg)

        async with self._sem.get():
            t0 = time.perf_counter()
            try:
                res = await retry_async(
                    lambda: self._call(messages, system, max_tokens, hints),
                    max_retries=self.cfg.max_retries,
                    what=f"{self.cfg.provider}:{self.cfg.model}",
                )
            except ProviderError as exc:
                self.ledger.add(
                    CallRecord(
                        role=role,
                        provider=self.cfg.provider,
                        model=self.cfg.model,
                        error=str(exc)[:500],
                        latency_s=time.perf_counter() - t0,
                    )
                )
                raise
            res.latency_s = time.perf_counter() - t0
        cost = price(self.cfg, res.prompt_tokens, res.completion_tokens)
        extra = dict(res.extra)
        if not self.supports_sampling_controls:
            extra["sampling_controls"] = "unsupported"  # D-09
        self.ledger.add(
            CallRecord(
                role=role,
                provider=self.cfg.provider,
                model=res.model,
                prompt_tokens=res.prompt_tokens,
                completion_tokens=res.completion_tokens,
                latency_s=res.latency_s,
                cost_usd=cost,
                cost_is_estimate=cost is None,
                tokens_are_estimate=res.tokens_are_estimate,
                extra=extra,
            )
        )
        if self.cache is not None:
            self.cache.set(key, {k: v for k, v in res.__dict__.items() if k != "cached"})
        return res

    async def _call(
        self,
        messages: list[Message],
        system: str | None,
        max_tokens: int,
        hints: dict[str, Any] | None,
    ) -> LLMResult:
        raise NotImplementedError

    async def aclose(self) -> None:
        return None


# ------------------------------------------------------------------------------- Anthropic


class AnthropicLLM(BaseLLM):
    """Claude via the official `anthropic` SDK (async client).

    Current Claude models reject `temperature`/`top_p`/`top_k`; nothing is sent for them (D-09).
    Depth/cost is controlled with `output_config.effort` (`params.effort`, default "low").
    Server-side model fallbacks are never enabled implicitly (D-11).
    """

    supports_sampling_controls = False

    def __init__(self, cfg: ProviderConfig, **kw: Any) -> None:
        super().__init__(cfg, **kw)
        import anthropic

        self._anthropic = anthropic
        key_env = cfg.api_key_env or "ANTHROPIC_API_KEY"
        self._client = anthropic.AsyncAnthropic(
            api_key=os.getenv(key_env) or None,
            timeout=cfg.timeout_s,
            max_retries=0,
            base_url=cfg.base_url,
        )

    async def _call(
        self,
        messages: list[Message],
        system: str | None,
        max_tokens: int,
        hints: dict[str, Any] | None,
    ) -> LLMResult:
        a = self._anthropic
        kwargs: dict[str, Any] = {
            "model": self.cfg.model,
            "max_tokens": max_tokens,
            "messages": messages,
            "output_config": {"effort": self.cfg.params.get("effort", "low")},
        }
        if system:
            kwargs["system"] = system
        try:
            resp = await self._client.messages.create(**kwargs)
        except a.RateLimitError as exc:
            ra = exc.response.headers.get("retry-after") if exc.response else None
            raise RetryableError(str(exc), retry_after=float(ra) if ra else None) from exc
        except (a.APITimeoutError, a.APIConnectionError, a.InternalServerError) as exc:
            raise RetryableError(str(exc)) from exc
        except a.AuthenticationError as exc:
            raise ProviderError("Anthropic authentication failed; set ANTHROPIC_API_KEY") from exc
        except a.APIStatusError as exc:
            if exc.status_code in (408, 409, 529) or exc.status_code >= 500:
                raise RetryableError(str(exc)) from exc
            raise ProviderError(f"Anthropic API error {exc.status_code}: {exc}") from exc
        if resp.stop_reason == "refusal":
            raise ProviderError(f"model refused (stop_reason=refusal): {resp.stop_details}")
        text = "".join(b.text for b in resp.content if b.type == "text")
        return LLMResult(
            text=text,
            model=resp.model,
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
            latency_s=0.0,
            stop_reason=resp.stop_reason,
        )

    async def aclose(self) -> None:
        await self._client.close()


# ------------------------------------------------------------------------- OpenAI-compatible


class OpenAICompatLLM(BaseLLM):
    """Any `/v1/chat/completions` server: OpenAI, vLLM, Ollama, llama.cpp, DeepSeek, etc."""

    def __init__(self, cfg: ProviderConfig, **kw: Any) -> None:
        super().__init__(cfg, **kw)
        base = require_resolved(cfg.base_url, cfg.model) or os.getenv("OPENAI_COMPAT_BASE_URL")
        if not base:
            raise ConfigError(f"{cfg.model}: base_url (or OPENAI_COMPAT_BASE_URL) is required")
        key = os.getenv(cfg.api_key_env) if cfg.api_key_env else None
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._client = httpx.AsyncClient(
            base_url=base.rstrip("/"), headers=headers, timeout=cfg.timeout_s
        )

    async def _call(
        self,
        messages: list[Message],
        system: str | None,
        max_tokens: int,
        hints: dict[str, Any] | None,
    ) -> LLMResult:
        msgs = ([{"role": "system", "content": system}] if system else []) + messages
        body: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": msgs,
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": self.cfg.params.get("seed", 0),
        }
        body.update({k: v for k, v in self.cfg.params.items() if k != "seed"})
        try:
            r = await self._client.post("/chat/completions", json=body)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise RetryableError(f"{type(exc).__name__}: {exc}") from exc
        if r.status_code != 200:
            raise http_error_to_exception(r.status_code, r.text, r.headers.get("retry-after"))
        data = r.json()
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"malformed chat completion: {str(data)[:300]}") from exc
        usage = data.get("usage") or {}
        estimate = "prompt_tokens" not in usage
        pt = usage.get("prompt_tokens") or sum(self._tok.count(m["content"]) for m in msgs)
        ct = usage.get("completion_tokens") or self._tok.count(text)
        return LLMResult(
            text=text,
            model=data.get("model", self.cfg.model),
            prompt_tokens=pt,
            completion_tokens=ct,
            latency_s=0.0,
            stop_reason=data["choices"][0].get("finish_reason"),
            tokens_are_estimate=estimate,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


# ------------------------------------------------------------------------------- Offline

_STOP = frozenset(
    """a an the of in on at to for from by with and or but is are was were be been being do
    does did what which who whom whose when where why how that this these those it its as into
    than then there their they them he she his her you your i we our not no""".split()
)
_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_CAP_PHRASE = re.compile(
    r"\b([A-Z][a-zA-Z0-9\-']+(?:\s+(?:of|de|the|and)?\s*[A-Z][a-zA-Z0-9\-']+)*)"
)
INSUFFICIENT = "insufficient information"


class OfflineLLM(BaseLLM):
    """Deterministic, dependency-free stand-in for an LLM (D-18). NOT a research model.

    Handles: LightRAG entity extraction / gleaning / keyword / summary prompts, extractive answer
    generation with citations and refusal, router classification and reference-match judging.
    """

    async def _call(
        self,
        messages: list[Message],
        system: str | None,
        max_tokens: int,
        hints: dict[str, Any] | None,
    ) -> LLMResult:
        prompt = messages[-1]["content"] if messages else ""
        hints = hints or {}
        task = hints.get("task")
        if task == "generate":
            text = extractive_answer(hints["question"], hints["passages"])
        elif task == "route":
            text = json.dumps({"query_type": hints.get("rule_type", "single_hop")})
        elif task == "judge":
            text = json.dumps({"verdict": offline_judge(hints["answer"], hints["references"])})
        elif task == "pairwise":
            text = json.dumps({"winner": "tie"})
        elif "high_level_keywords" in prompt and "User Query:" in prompt:
            text = _offline_keywords(prompt)
        elif "---Input Text---" in prompt and "Based on the last extraction" not in prompt:
            text = heuristic_extract(_between(prompt, "---Input Text---\n```\n", "\n```"))
        elif "Based on the last extraction" in prompt:
            text = "<|COMPLETE|>"  # gleaning pass: nothing more to add
        elif "Description List" in prompt:
            text = " ".join(dict.fromkeys(re.findall(r'"description":\s*"([^"]+)"', prompt)))
        else:
            text = INSUFFICIENT
        text = self._tok.truncate(text, max_tokens)
        pt = sum(self._tok.count(m["content"]) for m in messages) + (
            self._tok.count(system) if system else 0
        )
        return LLMResult(
            text=text,
            model=f"offline/{self.cfg.model}",
            prompt_tokens=pt,
            completion_tokens=self._tok.count(text),
            latency_s=0.0,
        )


def _between(text: str, start: str, end: str) -> str:
    i = text.rfind(start)
    if i < 0:
        return text
    j = text.find(end, i + len(start))
    return text[i + len(start) : j if j >= 0 else None]


def _offline_keywords(prompt: str) -> str:
    query = _between(prompt, "User Query:", "\n").strip()
    caps = [
        m.group(1)
        for m in _CAP_PHRASE.finditer(query)
        if m.group(1).lower() not in _STOP and m.start() > 0
    ]
    content = [w for w in word_tokens(query) if w not in _STOP and len(w) > 2]
    return json.dumps(
        {
            "high_level_keywords": content[:5] or [query[:50]],
            "low_level_keywords": list(dict.fromkeys(caps))[:8] or content[:5],
        }
    )


def heuristic_extract(text: str) -> str:
    """Capitalised-phrase entities, sentence co-occurrence relations, in LightRAG's format."""
    entities: dict[str, str] = {}
    relations: list[tuple[str, str, str]] = []
    for sent in _SENT_RE.split(text.strip()):
        names = []
        for m in _CAP_PHRASE.finditer(sent):
            name = m.group(1).strip()
            if name.lower() in _STOP or len(name) < 3:
                continue
            names.append(name)
            entities.setdefault(name, sent.strip()[:300])
        uniq = list(dict.fromkeys(names))[:6]
        for i in range(len(uniq) - 1):
            relations.append((uniq[i], uniq[i + 1], sent.strip()[:300]))
    lines = [f"entity<|#|>{n}<|#|>concept<|#|>{d}" for n, d in list(entities.items())[:40]]
    lines += [f"relation<|#|>{a}<|#|>{b}<|#|>co-occurrence<|#|>{d}" for a, b, d in relations[:60]]
    return "\n".join([*lines, "<|COMPLETE|>"])


def extractive_answer(question: str, passages: list[str], min_overlap: float = 0.34) -> str:
    """Pick the passage sentence with the best IDF-weighted overlap with the question; cite it.
    Refuse when less than `min_overlap` of the question's content weight is covered."""
    q_terms = {w for w in word_tokens(question) if w not in _STOP}
    if not q_terms or not passages:
        return INSUFFICIENT
    df: Counter[str] = Counter()
    sents: list[tuple[int, str, set[str]]] = []
    for i, p in enumerate(passages, start=1):
        for s in _SENT_RE.split(p):
            toks = set(word_tokens(s))
            if toks:
                sents.append((i, s.strip(), toks))
                df.update(toks)
    n = len(sents) or 1
    idf = {t: math.log(1 + n / (1 + df[t])) for t in q_terms}
    total = sum(idf.values())
    best = max(sents, key=lambda x: (sum(idf[t] for t in q_terms & x[2]), -x[0]), default=None)
    if best is None:
        return INSUFFICIENT
    score = sum(idf[t] for t in q_terms & best[2]) / total
    if score < min_overlap:
        return INSUFFICIENT
    return f"{best[1]} [{best[0]}]"


def offline_judge(answer: str, references: list[str]) -> str:
    from elevate_rag.evaluation.metrics import normalize_answer, token_f1

    na = normalize_answer(answer)
    if not references:
        return "wrong"
    if any(normalize_answer(r) and normalize_answer(r) in na for r in references):
        return "correct"
    best = max(token_f1(answer, r) for r in references)
    return "partial" if best >= 0.3 else "wrong"


# ------------------------------------------------------------------------------- factory


def make_llm(
    cfg: ProviderConfig, *, role: str, ledger: UsageLedger, cache: KVCache | None = None
) -> BaseLLM:
    providers: dict[str, type[BaseLLM]] = {
        "anthropic": AnthropicLLM,
        "openai-compatible": OpenAICompatLLM,
        "offline": OfflineLLM,
    }
    try:
        cls = providers[cfg.provider]
    except KeyError:
        raise ConfigError(
            f"unknown LLM provider {cfg.provider!r}; choose one of {sorted(providers)}"
        ) from None
    return cls(cfg, role=role, ledger=ledger, cache=cache)
