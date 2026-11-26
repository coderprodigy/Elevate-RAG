"""Error taxonomy for Elevate-RAG.

Every user-facing failure derives from :class:`ElevateError`. Each subclass takes the
structured facts its requirement asks the message to name (file, field, backend, model,
...) as keyword arguments, keeps them as attributes for programmatic handling, and builds
an actionable message from them. Callers must never pass secret values into these errors.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

SETUP_DOC = "docs/SETUP.md"
DATASETS_DOC = "docs/DATASETS.md"


def _rebuild(cls: type[ElevateError], kwargs: dict[str, Any]) -> ElevateError:
    return cls(**kwargs)


class ElevateError(Exception):
    """Base class for all Elevate-RAG errors.

    Subclasses take keyword-only arguments and record them with :meth:`_keep`, so the
    error survives pickling (e.g. when raised inside a multiprocessing worker).
    """

    _init_kwargs: dict[str, Any] | None = None

    def _keep(self, **kwargs: Any) -> None:
        self._init_kwargs = kwargs

    def __reduce__(self) -> Any:
        if self._init_kwargs is None:
            return super().__reduce__()
        return (_rebuild, (type(self), self._init_kwargs))


class ConfigError(ElevateError):
    """Invalid configuration: names the file, the field and the expected type (Req 1.3, 3.8)."""

    def __init__(
        self,
        *,
        file: str | Path,
        field: str,
        expected: str,
        detail: str | None = None,
    ) -> None:
        self._keep(file=file, field=field, expected=expected, detail=detail)
        self.file = str(file)
        self.field = field
        self.expected = expected
        self.detail = detail
        message = f"Invalid configuration in {self.file}: field '{field}' expected {expected}"
        if detail:
            message += f" ({detail})"
        super().__init__(message)


class BackendError(ElevateError):
    """A database backend failed its health check; points at the Setup_Docs (Req 2.6)."""

    def __init__(self, *, backend: str, reason: str, setup_section: str | None = None) -> None:
        self._keep(backend=backend, reason=reason, setup_section=setup_section)
        self.backend = backend
        self.reason = reason
        self.setup_section = setup_section or backend.lower()
        self.doc_ref = f"{SETUP_DOC}#{self.setup_section}"
        super().__init__(
            f"Backend '{backend}' failed its health check: {reason}. See {self.doc_ref}."
        )


class ModelCallError(ElevateError):
    """A model call still failed after all retries (Req 3.5)."""

    def __init__(self, *, model: str, reason: str, attempts: int) -> None:
        self._keep(model=model, reason=reason, attempts=attempts)
        self.model = model
        self.reason = reason
        self.attempts = attempts
        super().__init__(
            f"Model call to '{model}' failed after {attempts} attempt(s); last failure: {reason}"
        )


class BudgetExceeded(ElevateError):  # noqa: N818 - name fixed by the design
    """Accumulated cost reached the Spending_Cap (Req 4.4)."""

    def __init__(self, *, spent_usd: float, cap_usd: float, remaining_items: int) -> None:
        self._keep(spent_usd=spent_usd, cap_usd=cap_usd, remaining_items=remaining_items)
        self.spent_usd = spent_usd
        self.cap_usd = cap_usd
        self.remaining_items = remaining_items
        super().__init__(
            f"Spending cap reached: spent ${spent_usd:.4f} of ${cap_usd:.4f}; "
            f"{remaining_items} item(s) remaining. Completed results were kept; "
            "rerun to resume from cache."
        )


class DatasetError(ElevateError):
    """A raw dataset file is missing or has an unexpected schema (Req 6.9)."""

    def __init__(
        self,
        *,
        file: str | Path,
        dataset_section: str,
        missing_field: str | None = None,
        detail: str | None = None,
    ) -> None:
        self._keep(
            file=file, dataset_section=dataset_section, missing_field=missing_field, detail=detail
        )
        self.file = str(file)
        self.missing_field = missing_field
        self.dataset_section = dataset_section
        self.doc_ref = f"{DATASETS_DOC}#{dataset_section}"
        if missing_field is None:
            message = f"Dataset file {self.file} is missing or unreadable"
        else:
            message = f"Dataset file {self.file} is missing required field '{missing_field}'"
        if detail:
            message += f" ({detail})"
        super().__init__(f"{message}. See {self.doc_ref}.")


class EmbeddingMismatchError(ElevateError):
    """Query embedder differs from the embedder stored with the index (Req 11.2)."""

    def __init__(self, *, query_model: str, index_model: str) -> None:
        self._keep(query_model=query_model, index_model=index_model)
        self.query_model = query_model
        self.index_model = index_model
        super().__init__(
            f"Embedding model mismatch: query embedder '{query_model}' "
            f"differs from index embedder '{index_model}'"
        )


class TuningOnTestError(ElevateError):
    """A tuning operation was requested on a non-dev split (Req 7.3, 14.3)."""

    def __init__(self, *, operation: str, split: str, allowed: Sequence[str] = ("dev",)) -> None:
        self._keep(operation=operation, split=split, allowed=allowed)
        self.operation = operation
        self.split = split
        self.allowed = tuple(allowed)
        super().__init__(
            f"Tuning operation '{operation}' requested on split '{split}': "
            f"tuning is restricted to {', '.join(self.allowed)}"
        )


__all__ = [
    "DATASETS_DOC",
    "SETUP_DOC",
    "BackendError",
    "BudgetExceeded",
    "ConfigError",
    "DatasetError",
    "ElevateError",
    "EmbeddingMismatchError",
    "ModelCallError",
    "TuningOnTestError",
]
