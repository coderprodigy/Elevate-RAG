"""Tracer: one record per LLM call, to Langfuse or a local JSONL log (Req 4.6, 4.7).

:func:`make_tracer` picks the backend. With both ``LANGFUSE_PUBLIC_KEY`` and
``LANGFUSE_SECRET_KEY`` configured and the optional ``langfuse`` package installed
(``uv sync --extra tracing``), traces go to Langfuse (``LANGFUSE_HOST``, default Langfuse
Cloud). Otherwise each trace is appended as one JSON line to
``<trace_dir>/<run_id>.jsonl``.

If a Langfuse send fails, the trace is written to the local JSONL log instead, so a network
problem never loses a trace or stops a Run. Trace records never contain API keys.
"""

from __future__ import annotations

import importlib
import json
import logging
import threading
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from elevate.config.schema import Endpoints, ExperimentConfig, Secrets

logger = logging.getLogger(__name__)

TRACE_SUFFIX = ".jsonl"


@dataclass(frozen=True)
class LLMTrace:
    """One LLM or embedding call as seen by the LLM_Gateway."""

    run_id: str
    role: str
    model_key: str
    model_id: str
    input: Any
    output: Any
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    cached: bool = False
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["metadata"] = dict(self.metadata)
        return data


class Tracer(Protocol):
    """Trace sink used by the LLM_Gateway (design: Gateway section)."""

    def trace(self, call: LLMTrace) -> None: ...

    def flush(self) -> None: ...


class JSONLTracer:
    """Appends each trace as one JSON line; safe to share between threads (Req 4.7)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def trace(self, call: LLMTrace) -> None:
        line = json.dumps(call.to_dict(), ensure_ascii=False, sort_keys=True, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def flush(self) -> None:
        """Every write is already flushed on close; nothing to do."""

    def read(self) -> list[dict[str, Any]]:
        """All traces written so far, oldest first."""
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]


class LangfuseTracer:
    """Sends each trace to Langfuse as a ``generation`` observation (Req 4.6).

    ``client`` is a ``langfuse.Langfuse`` instance (or anything with the same
    ``start_observation``/``flush`` methods). Failed sends go to ``fallback``.
    """

    def __init__(self, client: Any, *, fallback: JSONLTracer) -> None:
        self.client = client
        self.fallback = fallback

    def trace(self, call: LLMTrace) -> None:
        metadata = {
            **dict(call.metadata),
            "run_id": call.run_id,
            "role": call.role,
            "model_key": call.model_key,
            "cached": call.cached,
            "latency_s": call.latency_s,
            "timestamp": call.timestamp,
        }
        try:
            observation = self.client.start_observation(
                name=f"{call.role}:{call.model_key}",
                as_type="generation",
                model=call.model_id,
                input=call.input,
                output=call.output,
                metadata=metadata,
                usage_details={
                    "input": call.prompt_tokens,
                    "output": call.completion_tokens,
                },
                cost_details={"total": call.cost_usd},
                level="ERROR" if call.error else "DEFAULT",
                status_message=call.error,
            )
            observation.end()
        except Exception as exc:  # tracing must never stop a Run
            logger.warning(
                "Langfuse trace failed (%s); writing to %s", type(exc).__name__, self.fallback.path
            )
            self.fallback.trace(call)

    def flush(self) -> None:
        try:
            self.client.flush()
        except Exception as exc:
            logger.warning("Langfuse flush failed: %s", type(exc).__name__)


def langfuse_configured(secrets: Secrets) -> bool:
    """True when both Langfuse keys are set to non-empty values."""
    public, secret = secrets.langfuse_public_key, secrets.langfuse_secret_key
    return bool(
        public is not None
        and secret is not None
        and public.get_secret_value()
        and secret.get_secret_value()
    )


def _default_langfuse_client(secrets: Secrets, endpoints: Endpoints) -> Any:
    """Build a ``langfuse.Langfuse`` client; raises ``ImportError`` if not installed."""
    module = importlib.import_module("langfuse")
    assert secrets.langfuse_public_key is not None
    assert secrets.langfuse_secret_key is not None
    kwargs: dict[str, Any] = {
        "public_key": secrets.langfuse_public_key.get_secret_value(),
        "secret_key": secrets.langfuse_secret_key.get_secret_value(),
    }
    if endpoints.langfuse_host:
        kwargs["host"] = endpoints.langfuse_host
    return module.Langfuse(**kwargs)


ClientFactory = Callable[[Secrets, Endpoints], Any]


def make_tracer(
    config: ExperimentConfig,
    run_id: str,
    *,
    client_factory: ClientFactory | None = None,
) -> JSONLTracer | LangfuseTracer:
    """Langfuse tracer when credentials exist and the package loads, else local JSONL."""
    local = JSONLTracer(Path(config.paths.trace_dir) / f"{run_id}{TRACE_SUFFIX}")
    if not langfuse_configured(config.secrets):
        return local
    factory = client_factory or _default_langfuse_client
    try:
        client = factory(config.secrets, config.endpoints)
    except ImportError:
        logger.warning(
            "Langfuse keys are set but the 'langfuse' package is not installed "
            "(uv sync --extra tracing); writing traces to %s",
            local.path,
        )
        return local
    except Exception as exc:
        logger.warning(
            "Langfuse client could not start (%s); writing traces to %s",
            type(exc).__name__,
            local.path,
        )
        return local
    return LangfuseTracer(client, fallback=local)


__all__ = [
    "TRACE_SUFFIX",
    "JSONLTracer",
    "LLMTrace",
    "LangfuseTracer",
    "Tracer",
    "langfuse_configured",
    "make_tracer",
]
