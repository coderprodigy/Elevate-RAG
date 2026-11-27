"""Optional metadata-only tracing to a local Langfuse instance."""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from types import TracebackType
from typing import Any
from urllib.parse import urlsplit


class LocalTracing:
    def __init__(self, enabled: bool) -> None:
        self.client: Any = None
        if not enabled:
            return
        base = os.environ.get("LANGFUSE_BASE_URL", "")
        url = urlsplit(base)
        if (
            url.scheme not in {"http", "https"}
            or url.hostname not in {"localhost", "127.0.0.1", "::1"}
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("Langfuse tracing requires an explicit loopback LANGFUSE_BASE_URL")
        public, secret = (
            os.environ.get("LANGFUSE_PUBLIC_KEY"),
            os.environ.get("LANGFUSE_SECRET_KEY"),
        )
        if not public or not secret:
            raise ValueError("local Langfuse requires public and secret project keys")
        from langfuse import Langfuse

        self.client = Langfuse(
            public_key=public,
            secret_key=secret,
            base_url=base,
            environment="elevate-rag",
            timeout=5,
        )

    @contextmanager
    def observe(
        self,
        name: str,
        metadata: dict[str, Any],
        *,
        kind: str = "span",
        model: str | None = None,
    ) -> Iterator[Any]:
        if self.client is None:
            yield None
            return
        failure: tuple[BaseException, TracebackType | None] | None = None
        with self.client.start_as_current_observation(
            name=name,
            as_type=kind,
            metadata=metadata,
            model=model,
        ) as span:
            try:
                yield span
            except BaseException as exc:
                # Exit without handing the exporter raw exception text or stack frames.
                span.update(level="ERROR", status_message=type(exc).__name__)
                failure = (exc, exc.__traceback__)
        if failure is not None:
            raise failure[0].with_traceback(failure[1])

    def flush(self) -> None:
        if self.client is not None:
            self.client.flush()
