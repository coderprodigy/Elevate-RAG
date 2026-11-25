"""Resumable, budget-aware concurrent processing that appends one JSON line per finished item.

Used by the answer runner, the judge, and faithfulness scoring. The caller filters out items that
are already done (by reading the output file), so re-running continues where it stopped.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from hgrag.llm.budget import BudgetExceededError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchOutcome:
    n_new: int
    n_failed: int
    stopped_reason: str | None  # set when a budget cap stopped the batch (resumable)


async def run_resumable[T](
    items: Sequence[T],
    worker: Callable[[T], Awaitable[BaseModel]],
    out_path: Path,
    *,
    concurrency: int = 8,
    label: Callable[[T], str] = str,
) -> BatchOutcome:
    """Run ``worker`` on every item; append each result to ``out_path`` as soon as it finishes.

    A budget stop ends the batch cleanly. Any other error skips that item (logged), so one bad
    item never kills a long run, and the next resume retries it.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)
    stop = asyncio.Event()
    stopped: list[str] = []
    failures = 0

    with out_path.open("a", encoding="utf-8") as out:

        async def process(item: T) -> int:
            nonlocal failures
            async with semaphore:
                if stop.is_set():
                    return 0
                try:
                    result = await worker(item)
                except BudgetExceededError as error:
                    stop.set()
                    stopped.append(str(error))
                    return 0
                except Exception:
                    logger.exception("item %s failed", label(item))
                    failures += 1
                    return 0
                out.write(result.model_dump_json() + "\n")
                out.flush()
                return 1

        written = await asyncio.gather(*(process(item) for item in items))

    reason = stopped[0] if stopped else None
    if reason:
        logger.warning("stopped early (resumable): %s", reason)
    return BatchOutcome(n_new=sum(written), n_failed=failures, stopped_reason=reason)


def read_models[M: BaseModel](path: Path, model: type[M]) -> list[M]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [model.model_validate_json(line) for line in handle if line.strip()]
