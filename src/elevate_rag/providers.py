"""Model adapters, content-addressed caches and explicit API spend accounting."""

import asyncio
import json
import math
import os
import re
import sqlite3
import time
from collections import Counter
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from numpy.typing import NDArray

from .config import Model, Settings
from .data import CharacterTokenizer, HuggingFaceTokenizer, Tokenizer, stable_hash, write_json
from .tracing import LocalTracing

Vector = NDArray[np.float32]
REFUSAL = "insufficient information"


def terms(text: str) -> list[str]:
    return re.findall(r"[^\W_]+(?:[-'][^\W_]+)*", text.casefold())


def normalized_vectors(vectors: Any, rows: int, dimensions: int) -> Vector:
    array = np.asarray(vectors, dtype=np.float32)
    if array.shape != (rows, dimensions) or not np.isfinite(array).all():
        raise ValueError(f"invalid embedding matrix {array.shape}, expected {(rows, dimensions)}")
    lengths = np.linalg.norm(array.astype(np.float64), axis=1, keepdims=True)
    return np.asarray(array / np.maximum(lengths, np.finfo(np.float32).tiny), dtype=np.float32)


@dataclass(frozen=True, slots=True)
class Usage:
    role: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    estimated_usd: float | None
    cached: bool
    elapsed_seconds: float
    provider_reported: bool
    attempts: int
    called_at: str


class Cache:
    def __init__(self, path: Path):
        self.path = path

    def get(self, namespace: str, key: str) -> Any:
        path = self.path / namespace / f"{key}.json"
        try:
            row = json.loads(path.read_text())
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            raise ValueError(f"cache entry is unreadable: {path}") from exc
        if (
            not isinstance(row, dict)
            or "value" not in row
            or row.get("checksum") != stable_hash(row["value"])
        ):
            raise ValueError(f"cache checksum mismatch: {path}")
        return row["value"]

    def put(self, namespace: str, key: str, value: Any) -> None:
        write_json(
            self.path / namespace / f"{key}.json", {"checksum": stable_hash(value), "value": value}
        )


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    """Persistent reservations prevent concurrent calls from exceeding the used cap."""

    def __init__(self, path: Path, cap: float):
        self.path, self.cap = path, cap
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path)) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS charges (id INTEGER PRIMARY KEY, usd REAL NOT NULL)"
            )

    def reserve(self, maximum: float) -> int:
        if not math.isfinite(maximum) or maximum < 0:
            raise ValueError("budget reservation must be finite and nonnegative")
        with closing(sqlite3.connect(self.path, isolation_level=None)) as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                used = db.execute("SELECT COALESCE(SUM(usd), 0) FROM charges").fetchone()[0]
                if self.cap <= 0 or used + maximum > self.cap:
                    raise BudgetExceeded(f"project API budget exhausted (cap USD {self.cap:.4f})")
                handle = db.execute("INSERT INTO charges (usd) VALUES (?) RETURNING id", (maximum,))
                key = handle.fetchone()[0]
                db.commit()
            except BaseException:
                db.rollback()
                raise
        return int(key)

    def settle(self, key: int, actual: float) -> None:
        if not math.isfinite(actual) or actual < 0:
            raise ValueError("budget settlement must be finite and nonnegative")
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("UPDATE charges SET usd = ? WHERE id = ?", (actual, key))

    def used(self) -> float:
        with closing(sqlite3.connect(self.path)) as db:
            return float(db.execute("SELECT COALESCE(SUM(usd), 0) FROM charges").fetchone()[0])


class ProviderError(RuntimeError):
    pass


class Providers:
    def __init__(
        self, settings: Settings, cache: Cache, budget_path: Path, *, uncached: bool = False
    ):
        self.settings, self.cache = settings, cache
        self.budget = Budget(budget_path, settings.spending_cap)
        self.uncached = uncached
        self.semaphore = asyncio.Semaphore(settings.concurrency)
        self.tracing = LocalTracing(settings.langfuse_tracing)
        self.client = httpx.AsyncClient(timeout=settings.timeout, trust_env=False)
        self.usage: list[Usage] = []
        self.local: dict[str, Any] = {}
        self.tokenizers: dict[str, Tokenizer] = {}

    async def __aenter__(self) -> "Providers":
        return self

    async def __aexit__(self, *error: object) -> None:
        try:
            await self.client.aclose()
        finally:
            if self.tracing.client is not None:
                await asyncio.to_thread(self.tracing.flush)

    def tokenizer(self, *, context: bool) -> Tokenizer:
        if self.settings.synthetic:
            return CharacterTokenizer()
        prefix = "context" if context else "chunk"
        if prefix not in self.tokenizers:
            self.tokenizers[prefix] = HuggingFaceTokenizer(
                getattr(self.settings, f"{prefix}_tokenizer_model"),
                getattr(self.settings, f"{prefix}_tokenizer_revision"),
                local_files_only=self.settings.local_models_only,
            )
        return self.tokenizers[prefix]

    def local_encoder(self, role: str, model: Model) -> Any:
        if role not in self.local:
            from sentence_transformers import CrossEncoder, SentenceTransformer

            cls = SentenceTransformer if role == "embedding" else CrossEncoder
            self.local[role] = cls(
                model.model,
                revision=model.revision,
                device=model.device,
                local_files_only=self.settings.local_models_only,
                trust_remote_code=False,
            )
        return self.local[role]

    @staticmethod
    def price(model: Model, prompt: int, completion: int) -> float:
        return (prompt * model.input_per_million + completion * model.output_per_million) / 1e6

    async def request(
        self, role: str, model: Model, path: str, payload: dict[str, Any]
    ) -> tuple[dict[str, Any], int]:
        headers = {}
        if model.key_env:
            if not (key := os.environ.get(model.key_env)):
                raise ProviderError(f"missing environment variable {model.key_env} for {role}")
            headers["Authorization"] = f"Bearer {key}"
        output_bound = model.max_output if path == "chat/completions" else 0
        batch = {
            "embeddings": len(payload.get("input", ())),
            "rerank": len(payload.get("documents", ())),
        }.get(path, 1)
        bound = self.price(model, model.context_window * batch, output_bound)
        for attempt in range(self.settings.retries + 1):
            async with self.semaphore:
                reservation = self.budget.reserve(bound) if model.paid else None
                try:
                    response = await self.client.post(
                        f"{model.base_url.rstrip('/')}/{path}",
                        json=payload,
                        headers=headers,
                    )
                except (httpx.ConnectError, httpx.ConnectTimeout):
                    if reservation is not None:
                        self.budget.settle(reservation, 0.0)
                    response = None
                except httpx.RequestError as exc:
                    raise ProviderError(f"{role} request failed: {type(exc).__name__}") from None
            if response is not None and response.is_success:
                try:
                    body = response.json()
                    if not isinstance(body, dict):
                        raise ValueError("expected object")
                    prompt, completion = self.usage_counts(body, path)
                except (TypeError, ValueError):
                    raise ProviderError(f"{role} returned invalid JSON or token usage") from None
                if reservation is not None and prompt is not None and completion is not None:
                    cost = self.price(model, prompt, completion)
                    self.budget.settle(reservation, cost)
                    if cost > bound:
                        raise BudgetExceeded(
                            f"{role} provider usage exceeded configured request bounds"
                        )
                return body, attempt + 1
            status = response.status_code if response is not None else None
            if status is not None and 400 <= status < 500 and reservation is not None:
                self.budget.settle(reservation, 0.0)
            if attempt == self.settings.retries or status not in {None, 429}:
                raise ProviderError(f"{role} request failed with {status or 'connection error'}")
            retry_after = response.headers.get("retry-after", "") if response is not None else ""
            delay = (
                min(30.0, float(retry_after))
                if re.fullmatch(r"\d+(?:\.\d+)?", retry_after)
                else 0.5 * 2**attempt
            )
            await asyncio.sleep(delay)
        raise AssertionError("retry loop exited unexpectedly")

    @staticmethod
    def usage_counts(body: dict[str, Any], path: str) -> tuple[int | None, int | None]:
        usage = body.get("usage") or {}
        if not isinstance(usage, dict):
            raise ValueError("usage must be an object")
        prompt = usage.get("prompt_tokens", usage.get("total_tokens") if path == "rerank" else None)
        completion = usage.get("completion_tokens", 0 if path in {"embeddings", "rerank"} else None)
        for value in (prompt, completion):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("usage counts must be nonnegative integers")
        return prompt, completion

    def record(
        self,
        role: str,
        model: Model,
        counts: tuple[int | None, int | None],
        started: float,
        *,
        cached: bool,
        reported: bool,
        attempts: int = 1,
    ) -> None:
        prompt, completion = counts
        price = (
            self.price(model, prompt, completion)
            if prompt is not None and completion is not None
            else None
        )
        if price is None and not model.paid:
            price = 0.0
        self.usage.append(
            Usage(
                role,
                model.identity,
                prompt,
                completion,
                0.0 if cached else price,
                cached,
                time.perf_counter() - started,
                reported,
                attempts,
                datetime.now(UTC).isoformat(),
            )
        )

    async def embed(self, texts: list[str], *, query: bool = False) -> Vector:
        model = self.settings.model("embedding")
        if not texts:
            return np.empty((0, model.dimensions), dtype=np.float32)
        started = time.perf_counter()
        inputs = list(dict.fromkeys(texts))
        identity = asdict(model)
        keys = [stable_hash("embedding-v1", identity, query, text) for text in inputs]
        values = [None if self.uncached else self.cache.get("embedding", key) for key in keys]
        if any(value is not None for value in values):
            self.record(
                "embedding", model, (None, 0), started, cached=True, reported=False, attempts=0
            )
        for start in range(0, len(inputs), self.settings.embedding_batch):
            indexes = [
                i
                for i in range(start, min(start + self.settings.embedding_batch, len(inputs)))
                if values[i] is None
            ]
            if not indexes:
                continue
            payload = [inputs[i] for i in indexes]
            if query and model.query_prompt:
                payload = [model.query_prompt + text for text in payload]
            counts: tuple[int | None, int | None] = (None, 0)
            attempts = 1
            if model.provider == "fixture":
                matrix = [self.fixture_embedding(text, model.dimensions) for text in payload]
            elif model.provider == "sentence-transformers":
                encoder = self.local_encoder("embedding", model)
                matrix = await asyncio.to_thread(encoder.encode, payload, show_progress_bar=False)
            else:
                body, attempts = await self.request(
                    "embedding",
                    model,
                    "embeddings",
                    {"model": model.model, "input": payload},
                )
                records = indexed_records(body, "data", len(payload))
                try:
                    matrix = [item["embedding"] for item in records]
                except KeyError:
                    raise ProviderError(
                        "embedding provider returned a record without a vector"
                    ) from None
                counts = self.usage_counts(body, "embeddings")
            checked = normalized_vectors(matrix, len(payload), model.dimensions)
            self.record(
                "embedding",
                model,
                counts,
                started,
                cached=False,
                reported=counts[0] is not None,
                attempts=attempts,
            )
            for index, row in zip(indexes, checked.tolist(), strict=True):
                values[index] = row
                if not self.uncached:
                    self.cache.put("embedding", keys[index], row)
        checked = normalized_vectors(values, len(inputs), model.dimensions)
        if len(inputs) == len(texts):
            return checked
        positions = {text: i for i, text in enumerate(inputs)}
        return checked[[positions[text] for text in texts]]

    @staticmethod
    def fixture_embedding(text: str, dimensions: int) -> Vector:
        vector = np.zeros(dimensions, dtype=np.float32)
        for token, frequency in Counter(terms(text)).items():
            digest = int(stable_hash(token)[:16], 16)
            vector[digest % dimensions] += frequency * (1 if (digest >> 32) & 1 else -1)
        return vector

    async def rerank(self, query: str, passages: list[str]) -> list[float]:
        model = self.settings.model("reranking")
        if not passages:
            return []
        started = time.perf_counter()
        counts: tuple[int | None, int | None] = (None, None)
        attempts = 1
        if model.provider == "fixture":
            query_tokens = set(terms(query))
            scores = [
                len(query_tokens.intersection(terms(passage))) / max(1, len(query_tokens))
                for passage in passages
            ]
        elif model.provider == "sentence-transformers":
            encoder = self.local_encoder("reranking", model)
            output = await asyncio.to_thread(encoder.predict, [(query, text) for text in passages])
            scores = [float(value) for value in output]
        elif model.provider == "jina-rerank":
            response, attempts = await self.request(
                "reranking",
                model,
                "rerank",
                {
                    "model": model.model,
                    "query": query,
                    "documents": passages,
                    "return_documents": False,
                },
            )
            ranked = indexed_records(response, "results", len(passages))
            try:
                scores = [float(item["relevance_score"]) for item in ranked]
            except (KeyError, TypeError, ValueError):
                raise ProviderError("reranking provider returned invalid scores") from None
            counts = self.usage_counts(response, "rerank")
        else:
            raise ValueError(
                "reranking requires a cross-encoder or the configured fixture provider"
            )
        if len(scores) != len(passages) or not np.isfinite(scores).all():
            raise ProviderError("reranker returned invalid scores")
        self.record(
            "reranking",
            model,
            counts,
            started,
            cached=False,
            reported=counts[0] is not None,
            attempts=attempts,
        )
        return scores

    async def chat(
        self, role: str, system: str, user: str, *, history: list[dict[str, str]] | None = None
    ) -> str:
        model = self.settings.model(role)
        messages = [
            {"role": "system", "content": system},
            *(history or []),
            {"role": "user", "content": user},
        ]
        key = stable_hash("chat-v1", role, asdict(model), messages)
        use_cache = not self.uncached and (
            role in {"extraction", "keywords"}
            or (role == "generation" and self.settings.cache_generation)
            or (role == "judge" and self.settings.cache_judge)
        )
        started = time.perf_counter()
        if use_cache and (cached := self.cache.get(role, key)) is not None:
            counts = tuple(cached["counts"])
            self.record(
                role, model, counts, started, cached=True, reported=cached["reported"], attempts=0
            )
            return str(cached["text"])
        if model.provider == "fixture":
            text = fixture_chat(role, user)
            counts = (sum(len(terms(message["content"])) for message in messages), len(terms(text)))
            attempts, reported = 1, False
        elif model.provider == "openai-compatible":
            payload: dict[str, Any] = {
                "model": model.model,
                "temperature": 0,
                "messages": messages,
                "max_tokens": model.max_output,
            }
            if model.seed is not None:
                payload["seed"] = model.seed
            body, attempts = await self.request(role, model, "chat/completions", payload)
            try:
                choice = body["choices"][0]
                text = choice["message"]["content"]
                if (
                    choice.get("finish_reason") in {"length", "content_filter"}
                    or not isinstance(text, str)
                    or not text.strip()
                ):
                    raise ValueError("empty, filtered or truncated response")
            except (KeyError, IndexError, TypeError, ValueError):
                raise ProviderError(f"{role} returned no complete assistant message") from None
            counts = self.usage_counts(body, "chat/completions")
            reported = counts[0] is not None and counts[1] is not None
        else:
            raise ValueError(f"{role}: this provider does not implement chat")
        self.record(
            role, model, counts, started, cached=False, reported=reported, attempts=attempts
        )
        if use_cache:
            self.cache.put(role, key, {"text": text, "counts": counts, "reported": reported})
        return text


def indexed_records(body: dict[str, Any], key: str, size: int) -> list[dict[str, Any]]:
    records = body.get(key)
    if not isinstance(records, list) or len(records) != size:
        raise ProviderError(f"provider returned invalid {key} rows")
    if any(not isinstance(row, dict) or type(row.get("index")) is not int for row in records):
        raise ProviderError(f"provider returned invalid {key} indexes")
    ordered = sorted(records, key=lambda row: row["index"])
    if [row["index"] for row in ordered] != list(range(size)):
        raise ProviderError(f"provider returned invalid {key} indexes")
    return ordered


def fixture_chat(role: str, user: str) -> str:
    """Deterministic provider for the explicitly synthetic fixture."""
    if role == "generation":
        data = json.loads(user)
        for number, evidence in enumerate(data["passages"], 1):
            for line in evidence["text"].splitlines():
                question, separator, answer = line.partition(" => ")
                if separator and question.casefold() == data["question"].casefold():
                    return f"{answer.strip()} [{number}]"
        return REFUSAL
    if role == "extraction" and user.lstrip().startswith("["):
        return "Synthetic thematic report for " + user[:300]
    if role == "extraction":
        names = sorted(set(re.findall(r"\b(?:Station|Beacon) [A-Z][a-z]+\b", user)))
        lines = [
            f"entity<|#|>{name}<|#|>artifact<|#|>{name} occurs in the synthetic passage."
            for name in names
        ]
        if len(names) >= 2:
            lines.append(
                f"relation<|#|>{names[0]}<|#|>{names[1]}<|#|>beacon,station<|#|>The beacon is associated with the station."
            )
        return "\n".join([*lines, "<|COMPLETE|>"])
    if role == "keywords":
        tokens = [token for token in terms(user) if len(token) > 3]
        return json.dumps({"high_level_keywords": tokens[:3], "low_level_keywords": tokens[:8]})
    if role == "router":
        return json.dumps({"route": "global" if "themes" in user else "V4"})
    raise ProviderError(f"no synthetic implementation for role {role}")
