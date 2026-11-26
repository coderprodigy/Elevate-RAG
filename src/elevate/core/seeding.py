"""Global random seeding for a Run (Requirement 1.6).

:func:`set_global_seeds` seeds Python's :mod:`random`, NumPy's legacy global generator and,
when it is installed, PyTorch (CPU and CUDA). It returns a :class:`SeedRecord` that the
Run_Record stores, so a reader can see exactly which frameworks were seeded and with what.

Torch is an optional extra (``elevate[ml]``), so it is imported lazily and skipped when absent.
"""

from __future__ import annotations

import importlib
import os
import random

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

MAX_SEED = 2**32 - 1  # NumPy's legacy global seed range is [0, 2**32 - 1]


class SeedRecord(BaseModel):
    """Seeds applied at Run start, keyed by framework."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int = Field(ge=0, le=MAX_SEED)
    python: int
    numpy: int
    torch: int | None = None
    """Seed given to torch, or ``None`` when torch is not installed."""
    pythonhashseed: str | None = None
    """Value of ``PYTHONHASHSEED`` in the environment (only settable before interpreter start)."""


def _seed_torch(seed: int) -> bool:
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return False
    torch.manual_seed(seed)
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and cuda.is_available():
        cuda.manual_seed_all(seed)
    return True


def set_global_seeds(seed: int) -> SeedRecord:
    """Seed Python, NumPy and torch (if installed) with ``seed`` and record what was set.

    Raises ``ValueError`` when ``seed`` is outside ``[0, 2**32 - 1]``.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an int, got {type(seed).__name__}")
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be in [0, {MAX_SEED}], got {seed}")

    random.seed(seed)
    np.random.seed(seed)  # seeds third-party code that uses NumPy's global RNG
    torch_seeded = _seed_torch(seed)

    return SeedRecord(
        seed=seed,
        python=seed,
        numpy=seed,
        torch=seed if torch_seeded else None,
        pythonhashseed=os.environ.get("PYTHONHASHSEED"),
    )


__all__ = ["MAX_SEED", "SeedRecord", "set_global_seeds"]
