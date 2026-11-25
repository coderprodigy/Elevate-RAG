"""Graph quality checks, run before any QA experiment (plan Phase 2 "Graph quality checks").

* Answer-entity coverage: share of short gold answers that appear as graph nodes. Han et al.
  report ~65% on HotpotQA/NQ; it caps graph-only accuracy. Gate G3 requires ≥ 60%.
* Graph shape (GraphRAG-Bench metrics): nodes, edges, average degree, clustering coefficient,
  isolated-node share, and the duplicate rate (distinct nodes whose names normalise identically).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import networkx as nx
from pydantic import BaseModel

from hgrag.data.schema import Query

GRAPH_FILE = "graph_chunk_entity_relation.graphml"
COVERAGE_GATE = 0.60
_MAX_ANSWER_WORDS = 6  # longer answers are sentences, not entities
_YES_NO = {"yes", "no"}
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_ARTICLES = re.compile(r"\b(the|a|an)\b")


class GraphShape(BaseModel):
    nodes: int
    edges: int
    average_degree: float
    clustering: float
    isolated_share: float
    duplicate_rate: float


class Coverage(BaseModel):
    eligible: int  # answerable questions with a short, non yes/no answer
    covered: int
    rate: float
    passes_gate: bool


def load_graph(working_dir: Path) -> Any:
    """The NetworkX graph a ``local``/``faiss`` backend build saved."""
    path = working_dir / GRAPH_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found (quality checks read NetworkX builds)")
    return nx.read_graphml(path)


def normalise_entity(name: str) -> str:
    return _NON_ALNUM.sub(" ", _ARTICLES.sub(" ", name.lower())).strip()


def graph_shape(graph: Any) -> GraphShape:
    undirected = nx.Graph(graph)
    n = undirected.number_of_nodes()
    names = Counter(normalise_entity(str(node)) for node in undirected.nodes)
    duplicates = sum(count - 1 for count in names.values() if count > 1)
    return GraphShape(
        nodes=n,
        edges=undirected.number_of_edges(),
        average_degree=round(2 * undirected.number_of_edges() / n, 3) if n else 0.0,
        clustering=round(nx.average_clustering(undirected), 4) if n else 0.0,
        isolated_share=round(nx.number_of_isolates(undirected) / n, 4) if n else 0.0,
        duplicate_rate=round(duplicates / n, 4) if n else 0.0,
    )


def answer_entity_coverage(graph: Any, queries: Iterable[Query]) -> Coverage:
    """A gold answer is covered if some node name equals it or contains it (normalised)."""
    nodes = {normalise_entity(str(node)) for node in graph.nodes}
    eligible = covered = 0
    for query in queries:
        answers = [normalise_entity(a) for a in query.answers]
        answers = [
            a for a in answers if a and a not in _YES_NO and len(a.split()) <= _MAX_ANSWER_WORDS
        ]
        if not query.answerable or not answers:
            continue
        eligible += 1
        covered += any(
            a in nodes or (len(a) >= 4 and any(a in node for node in nodes)) for a in answers
        )
    rate = covered / eligible if eligible else 0.0
    return Coverage(
        eligible=eligible, covered=covered, rate=round(rate, 4), passes_gate=rate >= COVERAGE_GATE
    )
