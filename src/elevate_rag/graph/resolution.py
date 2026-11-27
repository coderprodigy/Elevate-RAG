"""Entity resolution (FR-G3): normalised-name merge, then embedding-similarity merge.

Produces an auditable *merge plan* (groups of entity names + the evidence for each link) and
applies it through LightRAG's public `amerge_entities`, which rewires edges and updates the
vector stores. The plan is written to the index directory so merges can be reviewed and the
threshold ablated (A2).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from elevate_rag.providers.embeddings import BaseEmbedder

_ARTICLES = re.compile(r"^(the|a|an)\s+")
_NON_WORD = re.compile(r"[^\w\s]")


_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def normalize_entity(name: str) -> str:
    s = unicodedata.normalize("NFKC", name).lower().strip()
    s = _NON_WORD.sub(" ", s)
    return _ARTICLES.sub("", " ".join(s.split()))


def split_camel(name: str) -> str:
    """'HansenClarke' -> 'Hansen Clarke', '2004FrenchOpen' -> '2004 French Open'. Some extractors
    (Qwen2.5-7B in G1: 54 % of entities) drop the spaces from multi-word names."""
    return _CAMEL.sub(" ", name)


def entity_key(name: str) -> str:
    """Spacing-insensitive identity: 'UnitedKingdom', 'United Kingdom' and 'the united kingdom'
    share a key. Splitting camel case *before* normalising keeps article stripping working
    ('TheClickFive' == 'The Click Five'); dropping spaces afterwards keeps 'iPhone' == 'iphone'."""
    return normalize_entity(split_camel(name)).replace(" ", "")


class _UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {i: i for i in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # deterministic: lexicographically smaller root wins
            lo, hi = sorted((ra, rb))
            self.parent[hi] = lo


@dataclass
class MergePlan:
    groups: list[dict[str, Any]] = field(default_factory=list)  # {"target", "sources", "links"}

    @property
    def n_merged(self) -> int:
        return sum(len(g["sources"]) for g in self.groups)


def plan_merges(
    nodes: list[dict[str, Any]],
    embedder: BaseEmbedder | None,
    threshold: float,
    *,
    require_same_type: bool = True,
    block: int = 2048,
    max_merges: int = 10_000,
) -> MergePlan:
    """`nodes`: LightRAG node dicts (need `id`, optionally `entity_type`, `source_id`)."""
    names = sorted({n["id"] for n in nodes})
    if len(names) < 2:
        return MergePlan()
    meta = {n["id"]: n for n in nodes}
    uf = _UnionFind(names)
    links: list[tuple[str, str, str, float]] = []

    by_norm: dict[str, list[str]] = {}
    for n in names:
        by_norm.setdefault(entity_key(n), []).append(n)
    for group in by_norm.values():
        for other in group[1:]:
            uf.union(group[0], other)
            links.append((group[0], other, "normalized_name", 1.0))

    if embedder is not None and threshold < 1.0:
        vecs = embedder.embed(names, kind="document")
        types = [str(meta[n].get("entity_type", "")).lower() for n in names]
        for start in range(0, len(names), block):
            sims = vecs[start : start + block] @ vecs.T
            rows, cols = np.nonzero(sims >= threshold)
            for r, c in zip(rows, cols, strict=True):
                i = start + int(r)
                j = int(c)
                if j <= i:
                    continue
                if require_same_type and types[i] and types[j] and types[i] != types[j]:
                    continue
                if uf.find(names[i]) != uf.find(names[j]):
                    uf.union(names[i], names[j])
                    links.append((names[i], names[j], "embedding", float(sims[r, c])))

    members: dict[str, list[str]] = {}
    for n in names:
        members.setdefault(uf.find(n), []).append(n)

    def degree_key(n: str) -> tuple[int, int, str]:
        src = str(meta[n].get("source_id", ""))
        return (-(src.count("<SEP>") + 1 if src else 0), -len(n), n)

    plan = MergePlan()
    total = 0
    for root, group in sorted(members.items()):
        if len(group) < 2:
            continue
        target = sorted(group, key=degree_key)[0]
        sources = [g for g in group if g != target]
        if total + len(sources) > max_merges:
            break
        total += len(sources)
        glinks = [
            {"a": a, "b": b, "by": by, "score": round(s, 4)}
            for a, b, by, s in links
            if uf.find(a) == root
        ]
        plan.groups.append({"target": target, "sources": sources, "links": glinks})
    return plan
