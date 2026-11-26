"""Deterministic offline providers: scripted :class:`FakeProvider` and record-replay.

:class:`FakeProvider` answers in this order (Requirement 37.3):

1. the next item of its ``queue`` (a string is returned, an exception is raised), which lets
   tests script retries and malformed outputs call by call;
2. a :class:`FakeScript` lookup, the format of ``fixtures/tiny_sample/fake_llm``:

   - extraction: the first record whose ``match`` string (the document's first sentence)
     occurs in the prompt;
   - generation: the record for ``(metadata["question_id"], metadata["variant"])``, variant
     ``rag`` by default. ``[[doc_id]]`` markers become ``[n]``, the 1-based position of the
     document's first passage in ``metadata["context_doc_ids"]``; markers of documents not
     in the context are dropped, and when none of the cited documents is in the context
     the Refusal_String is returned. Without ``context_doc_ids`` all markers are dropped;
   - judge: the record for ``metadata["question_id"]`` whose prediction equals the
     normalised ``metadata["prediction"]``; otherwise normalised exact match against
     ``metadata["answers"]`` (or the Refusal_String when ``answerable`` is false);

3. a deterministic default derived from the request hash.

Embeddings are hash-based unit vectors. Every request is appended to ``calls`` so tests can
count provider calls (cache idempotence, Property 1).

:class:`RecordReplayProvider` wraps any provider: in ``record`` mode it stores each response
in a JSONL file keyed by the request hash; in ``replay`` mode it answers only from that file.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import string
import threading
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from elevate.core.ids import content_hash
from elevate.gateway.providers.base import (
    EmbeddingRequest,
    EmbeddingResponse,
    PermanentProviderError,
    Provider,
    ProviderRequest,
    ProviderResponse,
)

REFUSAL_STRING = "insufficient information"
EXTRACTION_COMPLETE = "<|COMPLETE|>"
DEFAULT_EMBEDDING_DIM = 64

_DOC_MARKER_RE = re.compile(r"\[\[([^\[\]]+)\]\]")
_NUM_MARKER_RE = re.compile(r"\[\d+\]")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,;:!?])")
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")


# --------------------------------------------------------------------------- text helpers
def _tidy(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)


def render_citations(template: str, context_doc_ids: Sequence[str] | None) -> str:
    """Replace ``[[doc_id]]`` with ``[n]`` by context order; drop markers of absent docs."""
    positions: dict[str, int] = {}
    for i, doc in enumerate(context_doc_ids or (), start=1):
        positions.setdefault(doc, i)

    def sub(match: re.Match[str]) -> str:
        n = positions.get(match.group(1))
        return f"[{n}]" if n is not None else ""

    return _tidy(_DOC_MARKER_RE.sub(sub, template))


def normalise_prediction(text: str) -> str:
    """Lower-case, citation markers removed, whitespace collapsed (judge script lookup)."""
    text = _DOC_MARKER_RE.sub("", _NUM_MARKER_RE.sub("", text))
    return _tidy(text).lower()


def normalise_answer(text: str) -> str:
    """SQuAD-style normalisation: lower-case, no punctuation, no articles, single spaces."""
    text = normalise_prediction(text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(_ARTICLES_RE.sub(" ", text).split())


def _judge_json(label: str, rationale: str) -> str:
    return json.dumps({"label": label, "rationale": rationale}, sort_keys=True)


def request_digest(request: ProviderRequest) -> str:
    """Content hash of every field of a completion request."""
    return content_hash(
        {
            "kind": "complete",
            "role": request.role,
            "model_id": request.model_id,
            "messages": [m.to_dict() for m in request.messages],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "seed": request.seed,
            "json_mode": request.json_mode,
            "metadata": dict(request.metadata),
        }
    )


def embedding_digest(request: EmbeddingRequest) -> str:
    return content_hash(
        {
            "kind": "embed",
            "role": request.role,
            "model_id": request.model_id,
            "texts": list(request.texts),
        }
    )


def hash_vector(text: str, dim: int = DEFAULT_EMBEDDING_DIM) -> list[float]:
    """Deterministic unit vector derived from SHA-256 of ``text``."""
    values: list[float] = []
    counter = 0
    while len(values) < dim:
        block = hashlib.sha256(f"{counter}:{text}".encode()).digest()
        values.extend((b / 127.5) - 1.0 for b in block)
        counter += 1
    values = values[:dim]
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


# --------------------------------------------------------------------------- scripts
def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@dataclass
class FakeScript:
    """Scripted responses in the ``fixtures/tiny_sample/fake_llm`` format."""

    extraction: list[dict[str, Any]] = field(default_factory=list)
    generation: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    judge: dict[tuple[str, str], str] = field(default_factory=dict)

    @classmethod
    def load(cls, directory: str | Path) -> FakeScript:
        """Load ``extraction.jsonl``, ``generation.jsonl`` and ``judge.jsonl`` (each optional)."""
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"fake LLM script directory not found: {directory}")
        extraction = _read_jsonl(directory / "extraction.jsonl")
        generation = {
            (row["question_id"], row["variant"]): row
            for row in _read_jsonl(directory / "generation.jsonl")
        }
        judge = {
            (row["question_id"], normalise_prediction(row["prediction"])): row["response"]
            for row in _read_jsonl(directory / "judge.jsonl")
        }
        return cls(extraction=extraction, generation=generation, judge=judge)

    # ----------------------------------------------------------------- lookups
    def extraction_response(self, prompt_text: str) -> str | None:
        for row in self.extraction:
            if row["match"] in prompt_text:
                return str(row["response"])
        return None

    def generation_response(self, metadata: Mapping[str, Any]) -> str | None:
        qid = metadata.get("question_id")
        if qid is None:
            return None
        row = self.generation.get((str(qid), str(metadata.get("variant", "rag"))))
        if row is None:
            return None
        context = metadata.get("context_doc_ids")
        cited: list[str] = list(row.get("cited_doc_ids") or [])
        if context is not None and cited and not set(cited) & set(context):
            return REFUSAL_STRING
        return render_citations(str(row["response_template"]), context)

    def judge_response(self, metadata: Mapping[str, Any]) -> str | None:
        qid = metadata.get("question_id")
        prediction = metadata.get("prediction")
        if qid is None or prediction is None:
            return None
        return self.judge.get((str(qid), normalise_prediction(str(prediction))))


def fallback_judge(metadata: Mapping[str, Any]) -> str:
    """Verdict by normalised exact match against ``answers`` or the Refusal_String."""
    prediction = normalise_answer(str(metadata.get("prediction", "")))
    answers = [str(a) for a in metadata.get("answers") or []]
    answerable = bool(metadata.get("answerable", bool(answers)))
    if not answerable:
        if prediction == normalise_answer(REFUSAL_STRING):
            return _judge_json("correct", "Correctly refuses an unanswerable question.")
        return _judge_json("wrong", "Answers a question the corpus cannot answer.")
    if any(prediction == normalise_answer(a) for a in answers):
        return _judge_json("correct", "Exact match with a reference answer.")
    return _judge_json("wrong", "No scripted verdict and no exact match with a reference answer.")


# --------------------------------------------------------------------------- providers
class FakeProvider:
    """Scripted, deterministic provider for offline tests and exit checks."""

    name = "fake"

    def __init__(
        self,
        script: FakeScript | None = None,
        *,
        queue: Iterable[str | BaseException] = (),
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    ) -> None:
        self.script = script or FakeScript()
        self._queue: deque[str | BaseException] = deque(queue)
        self.embedding_dim = embedding_dim
        self.calls: list[ProviderRequest] = []
        self.embed_calls: list[EmbeddingRequest] = []
        self._lock = threading.Lock()

    @classmethod
    def from_directory(cls, directory: str | Path, **kwargs: Any) -> FakeProvider:
        return cls(FakeScript.load(directory), **kwargs)

    @property
    def call_count(self) -> int:
        return len(self.calls) + len(self.embed_calls)

    def enqueue(self, *items: str | BaseException) -> None:
        with self._lock:
            self._queue.extend(items)

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        with self._lock:
            self.calls.append(request)
            item = self._queue.popleft() if self._queue else None
        if isinstance(item, BaseException):
            raise item
        text = item if item is not None else self._scripted(request)
        if text is None:
            text = self._default(request)
        return ProviderResponse(text=text, finish_reason="stop")

    def _scripted(self, request: ProviderRequest) -> str | None:
        if request.role == "extraction":
            return self.script.extraction_response(request.text)
        if request.role == "generator":
            return self.script.generation_response(request.metadata)
        if request.role == "judge":
            return self.script.judge_response(request.metadata)
        return None

    def _default(self, request: ProviderRequest) -> str:
        if request.role == "extraction":
            return EXTRACTION_COMPLETE
        if request.role == "generator" and not request.json_mode:
            return REFUSAL_STRING
        if request.role == "judge":
            return fallback_judge(request.metadata)
        tag = request_digest(request)[:12]
        if request.json_mode:
            return json.dumps({"fake_response": tag})
        return f"fake response {tag}"

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        with self._lock:
            self.embed_calls.append(request)
        vectors = [hash_vector(t, self.embedding_dim) for t in request.texts]
        return EmbeddingResponse(vectors=vectors)


ReplayMode = Literal["record", "replay", "auto"]


class RecordReplayProvider:
    """Records responses of ``inner`` to a JSONL file and replays them offline.

    Modes: ``record`` always calls ``inner`` and appends; ``replay`` answers only from the
    file and raises :class:`PermanentProviderError` on a miss; ``auto`` replays when a
    recording exists and records otherwise.
    """

    def __init__(
        self, path: str | Path, inner: Provider | None = None, *, mode: ReplayMode = "replay"
    ) -> None:
        if mode != "replay" and inner is None:
            raise ValueError(f"mode '{mode}' needs an inner provider to record from")
        self.path = Path(path)
        self.inner = inner
        self.mode = mode
        self.name = inner.name if inner is not None else "replay"
        self._lock = threading.Lock()
        self._records: dict[str, dict[str, Any]] = {}
        for row in _read_jsonl(self.path):
            self._records[row["key"]] = row["response"]

    def __len__(self) -> int:
        return len(self._records)

    def _lookup(self, key: str) -> dict[str, Any] | None:
        if self.mode == "record":
            return None
        found = self._records.get(key)
        if found is None and self.mode == "replay":
            raise PermanentProviderError(
                f"no recorded response for request {key[:12]} in {self.path}; "
                "re-record with mode='record'"
            )
        return found

    def _store(self, key: str, kind: str, response: dict[str, Any]) -> None:
        line = json.dumps({"key": key, "kind": kind, "response": response}, sort_keys=True)
        with self._lock:
            self._records[key] = response
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        key = request_digest(request)
        found = self._lookup(key)
        if found is not None:
            return ProviderResponse(**found)
        assert self.inner is not None
        response = self.inner.complete(request)
        self._store(key, "complete", asdict(response))
        return response

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        key = embedding_digest(request)
        found = self._lookup(key)
        if found is not None:
            return EmbeddingResponse(**found)
        assert self.inner is not None
        response = self.inner.embed(request)
        self._store(key, "embed", asdict(response))
        return response


__all__ = [
    "DEFAULT_EMBEDDING_DIM",
    "EXTRACTION_COMPLETE",
    "REFUSAL_STRING",
    "FakeProvider",
    "FakeScript",
    "RecordReplayProvider",
    "ReplayMode",
    "embedding_digest",
    "fallback_judge",
    "hash_vector",
    "normalise_answer",
    "normalise_prediction",
    "render_citations",
    "request_digest",
]
