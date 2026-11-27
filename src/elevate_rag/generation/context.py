"""Shared context assembly under one token budget (FR-C3, FR-G10, D-07).

Every retrieval mode passes through :func:`assemble_context`, so the generator sees comparable
context whichever path ran:

* chunk items are capped at `final_k` (same top-k for every mode);
* KG items (entities, relations, community reports) may use at most `kg_share` of the budget;
  KG budget left unused flows to chunks;
* an item that does not fit is truncated if ≥ `min_partial` tokens remain, else skipped.

The budget counts passage text only (the fixed system prompt and the question are identical for
all modes and are excluded).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from elevate_rag.types import ContextItem
from elevate_rag.utils import Tokenizer


@dataclass
class AssembledContext:
    items: list[ContextItem]
    tokens: int
    truncated: int
    dropped: int


def assemble_context(
    items: list[ContextItem],
    *,
    budget: int,
    final_k: int,
    kg_share: float,
    tokenizer: Tokenizer,
    min_partial: int = 64,
) -> AssembledContext:
    chunks = [it for it in items if it.kind == "chunk"][:final_k]
    kg = [it for it in items if it.kind != "chunk"]
    kg_cap = int(budget * kg_share) if chunks else budget
    out: list[ContextItem] = []
    used = truncated = dropped = 0

    def take(pool: list[ContextItem], cap: int) -> None:
        nonlocal used, truncated, dropped
        spent = 0
        for it in pool:
            n = tokenizer.count(it.text)
            room = min(cap - spent, budget - used)
            if n <= room:
                out.append(it)
                spent += n
                used += n
            elif room >= min_partial:
                out.append(
                    replace(
                        it,
                        text=tokenizer.truncate(it.text, room),
                        metadata={**it.metadata, "truncated": True},
                    )
                )
                spent += room
                used += room
                truncated += 1
            else:
                dropped += 1

    take(kg, kg_cap)
    take(chunks, budget)
    dropped += len([it for it in items if it.kind == "chunk"]) - len(chunks)
    return AssembledContext(out, used, truncated, dropped)
