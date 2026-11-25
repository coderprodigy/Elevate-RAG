"""Manual triple audit: two people label the same 50 random relations (correct/partial/wrong).

Each item shows the relation and the source text it was extracted from, so the labeller can
check it. Agreement is reported as Cohen's kappa (``hgrag.eval.human_labels.cohen_kappa``).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from hgrag.core.hashing import stable_hash

CHUNKS_FILE = "kv_store_text_chunks.json"
_SEP = "<SEP>"


class TripleItem(BaseModel):
    item_id: str
    source: str
    target: str
    keywords: str
    description: str
    evidence: str  # text of the chunk the relation came from


class TripleLabel(BaseModel):
    item_id: str
    labeller: str
    verdict: str  # correct | partial | wrong


def sample_triples(graph: Any, working_dir: Path, n: int, seed: int) -> list[TripleItem]:
    chunk_file = working_dir / CHUNKS_FILE
    chunks: dict[str, Any] = (
        json.loads(chunk_file.read_text(encoding="utf-8")) if chunk_file.is_file() else {}
    )
    edges = sorted(graph.edges(data=True), key=lambda e: (str(e[0]), str(e[1])))
    picked = random.Random(seed).sample(edges, min(n, len(edges)))
    items = []
    for source, target, data in picked:
        chunk_ids = str(data.get("source_id", "")).split(_SEP)
        evidence = next((chunks[c]["content"] for c in chunk_ids if c in chunks), "")
        items.append(
            TripleItem(
                item_id=stable_hash([source, target])[:12],
                source=str(source),
                target=str(target),
                keywords=str(data.get("keywords", "")),
                description=str(data.get("description", "")),
                evidence=evidence,
            )
        )
    return items


def audit_dir(working_dir: Path) -> Path:
    return working_dir / "audit"
