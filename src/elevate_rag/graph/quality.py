"""Graph quality checks (FR-G8) and the manual triple audit (FR-G9).

Run *before* any QA experiment. Gate G3: answer-entity coverage ≥ 60 % per dataset, plus an
audited triple sample with two annotators.
"""

from __future__ import annotations

import csv
import random
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx

from elevate_rag.graph.resolution import entity_key, normalize_entity, split_camel
from elevate_rag.indexing.chunk_store import ChunkStore
from elevate_rag.types import Question

_NON_ENTITY_ANSWERS = {"yes", "no", "insufficient information"}


def answer_entity_coverage(questions: list[Question], node_names: list[str]) -> dict[str, Any]:
    """Share of answerable questions whose (any) reference answer appears as a graph node.

    Matching (after :func:`normalize_entity`, node names also camel-split so space-stripped names
    like 'JaromeIginla' count): exact name match, or the answer occurs as a whole
    token sequence inside a node name ("Obama" ⊂ "barack obama"), or a node name of ≥ 2 tokens
    occurs inside a short (≤ 6-token) answer. Yes/no answers are not entities and are excluded
    from the denominator (reported separately).
    """
    # both the raw and the camel-split form of every node name, so 'JaromeIginla' matches
    norm_nodes = {normalize_entity(n) for n in node_names} | {
        normalize_entity(split_camel(n)) for n in node_names
    }
    padded = [f" {n} " for n in norm_nodes]
    multi = [n for n in norm_nodes if len(n.split()) >= 2]
    eligible = covered = excluded = 0
    by_type: dict[str, list[int]] = {}
    missed: list[str] = []
    for q in questions:
        if not q.answerable or not q.answers:
            continue
        answers = [normalize_entity(a) for a in q.answers if a.strip()]
        answers = [a for a in answers if a and a not in _NON_ENTITY_ANSWERS]
        if not answers:
            excluded += 1
            continue
        eligible += 1
        hit = False
        for a in answers:
            pa = f" {a} "
            if (
                a in norm_nodes
                or any(pa in n for n in padded)
                or (len(a.split()) <= 6 and any(f" {m} " in pa for m in multi))
            ):
                hit = True
                break
        covered += hit
        by_type.setdefault(q.query_type, [0, 0])
        by_type[q.query_type][0] += hit
        by_type[q.query_type][1] += 1
        if not hit and len(missed) < 25:
            missed.append(q.answers[0])
    return {
        "coverage": covered / eligible if eligible else None,
        "covered": covered,
        "eligible": eligible,
        "excluded_non_entity": excluded,
        "by_query_type": {t: v[0] / v[1] for t, v in by_type.items()},
        "missed_examples": missed,
    }


def graph_shape(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]], store: ChunkStore | None = None
) -> dict[str, Any]:
    g = nx.Graph()
    g.add_nodes_from(n["id"] for n in nodes)
    g.add_edges_from((e["source"], e["target"]) for e in edges)
    n, m = g.number_of_nodes(), g.number_of_edges()
    norm = Counter(entity_key(x["id"]) for x in nodes)
    dup_nodes = sum(c for c in norm.values() if c > 1)
    no_prov_nodes = [x["id"] for x in nodes if not x.get("chunk_ids")]
    no_prov_edges = [(e["source"], e["target"]) for e in edges if not e.get("chunk_ids")]
    out: dict[str, Any] = {
        "nodes": n,
        "edges": m,
        "avg_degree": (2 * m / n) if n else 0.0,
        "clustering_coefficient": nx.average_clustering(g) if n else 0.0,
        "connected_components": nx.number_connected_components(g) if n else 0,
        "isolated_nodes": sum(1 for _, d in g.degree() if d == 0),
        "duplicate_rate": dup_nodes / n if n else 0.0,
        "nodes_without_provenance": len(no_prov_nodes),
        "edges_without_provenance": len(no_prov_edges),
        "entity_types": dict(
            Counter(str(x.get("entity_type", "?")).lower() for x in nodes).most_common(15)
        ),
    }
    if store is not None and len(store):
        covered = {c for x in nodes for c in x.get("chunk_ids", [])}
        out["chunks_with_entities"] = len(covered) / len(store)
    return out


AUDIT_FIELDS = [
    "triple_id",
    "source",
    "relation",
    "target",
    "description",
    "evidence_chunk_id",
    "evidence_text",
    "label",
]
AUDIT_LABELS = ("correct", "partial", "wrong")


def export_triple_audit(
    edges: list[dict[str, Any]], store: ChunkStore, path: Path, n: int = 50, seed: int = 13
) -> int:
    """Write a CSV of `n` random triples with their evidence chunk; `label` left blank."""
    pool = sorted(edges, key=lambda e: (e["source"], e["target"]))
    sample = random.Random(seed).sample(pool, min(n, len(pool)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=AUDIT_FIELDS)
        w.writeheader()
        for i, e in enumerate(sample):
            cid = (e.get("chunk_ids") or [""])[0]
            w.writerow(
                {
                    "triple_id": f"t{i:03d}",
                    "source": e["source"],
                    "target": e["target"],
                    "relation": e.get("keywords", ""),
                    "description": e.get("description", ""),
                    "evidence_chunk_id": cid,
                    "evidence_text": store[cid].text[:1500] if cid in store else "",
                    "label": "",
                }
            )
    return len(sample)


def read_labels(path: Path, id_field: str, allowed: tuple[str, ...]) -> dict[str, str]:
    """Read a completed annotation CSV; rejects unknown or missing labels."""
    out = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            label = (row.get("label") or "").strip().lower()
            if label not in allowed:
                raise ValueError(
                    f"{path}: row {row[id_field]} has invalid/missing label "
                    f"{label!r}; allowed {allowed}"
                )
            out[row[id_field]] = label
    return out
