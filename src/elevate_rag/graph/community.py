"""Optional community layer (FR-G7, ablation A4 only): Leiden clustering + LLM report per
community, in the style of Microsoft GraphRAG.

Deviation from MS GraphRAG, documented in DECISIONS: instead of a map-reduce over *all* reports
at query time (≈331k tokens/query in GraphRAG-Bench), the top `top_reports` reports by embedding
similarity are added as `community` context items under the same shared token budget.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from elevate_rag.config import CommunityConfig
from elevate_rag.providers.embeddings import BaseEmbedder
from elevate_rag.providers.llm import BaseLLM
from elevate_rag.types import ContextItem
from elevate_rag.utils import Tokenizer

REPORT_SYSTEM = (
    "You write concise analytical reports about communities of entities in a "
    "knowledge graph. Use only the provided entities and relations."
)
REPORT_PROMPT = """Write a report (at most 250 words) describing this community: what it is about,
its key entities and how they relate. Start with a one-line title.

Entities:
{entities}

Relations:
{relations}
"""


def detect_communities(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]], cfg: CommunityConfig, seed: int = 13
) -> list[list[str]]:
    g = nx.Graph()
    g.add_nodes_from(n["id"] for n in nodes)
    for e in edges:
        g.add_edge(e["source"], e["target"], weight=float(e.get("weight") or 1.0))
    if g.number_of_edges() == 0:
        return []
    comms = nx.community.leiden_communities(
        g, weight="weight", resolution=cfg.resolution, metric="modularity", seed=seed
    )
    out = [sorted(c) for c in comms if len(c) >= cfg.min_size]
    return sorted(out, key=lambda c: (-len(c), c[0]))[: cfg.max_reports]


async def build_reports(
    communities: list[list[str]],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    llm: BaseLLM,
    max_input_tokens: int = 3000,
) -> list[dict[str, Any]]:
    tok = Tokenizer()
    by_id = {n["id"]: n for n in nodes}
    reports = []
    for i, members in enumerate(communities):
        mset = set(members)
        ent = "\n".join(
            f"- {m}: {str(by_id.get(m, {}).get('description', ''))[:300]}" for m in members[:40]
        )
        inner = [e for e in edges if e["source"] in mset and e["target"] in mset][:60]
        rel = "\n".join(
            f"- {e['source']} — {e['target']}: {str(e.get('description', ''))[:200]}" for e in inner
        )
        prompt = tok.truncate(REPORT_PROMPT.format(entities=ent, relations=rel), max_input_tokens)
        res = await llm.complete(
            [{"role": "user", "content": prompt}],
            system=REPORT_SYSTEM,
            max_tokens=400,
            task="community_report",
        )
        chunk_ids = sorted({c for m in members for c in by_id.get(m, {}).get("chunk_ids", [])})
        reports.append(
            {"id": f"community:{i}", "members": members, "report": res.text, "chunk_ids": chunk_ids}
        )
    return reports


class CommunityIndex:
    def __init__(self, reports: list[dict[str, Any]], vectors: np.ndarray) -> None:
        self.reports = reports
        self.vectors = vectors

    @classmethod
    def build(cls, reports: list[dict[str, Any]], embedder: BaseEmbedder) -> CommunityIndex:
        vecs = (
            embedder.embed([r["report"] for r in reports], kind="document")
            if reports
            else np.zeros((0, embedder.dimension), dtype=np.float32)
        )
        return cls(reports, vecs)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "community_reports.json").write_text(json.dumps(self.reports, indent=1))
        np.save(path / "community_vectors.npy", self.vectors)

    @classmethod
    def load(cls, path: Path) -> CommunityIndex | None:
        if not (path / "community_reports.json").exists():
            return None
        return cls(
            json.loads((path / "community_reports.json").read_text()),
            np.load(path / "community_vectors.npy"),
        )

    def search(self, query_vec: np.ndarray, k: int) -> list[ContextItem]:
        if not self.reports:
            return []
        scores = self.vectors @ query_vec.reshape(-1)
        order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))[:k]
        return [
            ContextItem(
                item_id=self.reports[i]["id"],
                kind="community",
                text=self.reports[i]["report"],
                score=float(scores[i]),
                source_chunk_ids=self.reports[i]["chunk_ids"][:20],
            )
            for i in order
        ]
