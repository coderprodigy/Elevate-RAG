"""NetworkX Graph_Store: the no-setup default graph backend (Req 2.2, 2.3).

The graph is an ``nx.MultiDiGraph`` keyed by relation, so two different relations between
the same pair of entities are kept apart. Each node stores its :class:`Entity` under the
``entity`` attribute and each edge its :class:`Relation` under ``relation``; neighbour
fetches run over an undirected view, matching the shared semantics in
:mod:`elevate.stores.graph.base`.

Persistence is a JSON file (GraphML cannot hold list attributes such as source chunk IDs
or profile embeddings without lossy encoding). :meth:`NetworkxGraphStore.save` writes
atomically through a temporary file, and :meth:`NetworkxGraphStore.load` restores an
identical store.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import networkx as nx

from elevate.core.errors import BackendError
from elevate.stores.graph.base import (
    Entity,
    GraphHealth,
    Relation,
    Subgraph,
    build_subgraph,
    check_hops,
    merge_entity,
    merge_relation,
    placeholder_node,
    resolve_name,
)

GRAPH_FILE_FORMAT = "elevate-graph"
GRAPH_FILE_VERSION = 1


class NetworkxGraphStore:
    """In-process graph backend built on NetworkX."""

    backend_name = "networkx"

    def __init__(self) -> None:
        self._graph: Any = nx.MultiDiGraph()

    # ------------------------------------------------------------------ writes

    def upsert_node(self, node: Entity) -> None:
        old = self._graph.nodes[node.name]["entity"] if node.name in self._graph else None
        self._graph.add_node(node.name, entity=merge_entity(old, node))

    def upsert_edge(self, edge: Relation) -> None:
        for endpoint in (edge.source, edge.target):
            if endpoint not in self._graph:
                self.upsert_node(placeholder_node(endpoint, edge.source_chunk_ids))
        old = self._edge(edge.source, edge.relation, edge.target)
        self._graph.add_edge(
            edge.source, edge.target, key=edge.relation, relation=merge_relation(old, edge)
        )

    # ------------------------------------------------------------------ reads

    def entity_by_name(self, name: str) -> Entity | None:
        found = resolve_name(name, self._graph.nodes, self._graph.nodes)
        return None if found is None else self._entity(found)

    def neighbours(self, name: str, hops: int) -> Subgraph:
        check_hops(hops)
        center = resolve_name(name, self._graph.nodes, self._graph.nodes)
        if center is None:
            return Subgraph(center=None, hops=hops)
        undirected = self._graph.to_undirected(as_view=True)
        distances: dict[str, int] = dict(
            nx.single_source_shortest_path_length(undirected, center, cutoff=hops)
        )
        incident = self._graph.subgraph(distances).edges(data="relation")
        return build_subgraph(
            center,
            hops,
            distances,
            {n: self._entity(n) for n in distances},
            (relation for _, _, relation in incident),
        )

    def health_check(self) -> GraphHealth:
        try:
            nodes, edges = self.node_count(), self.edge_count()
        except Exception as exc:  # pragma: no cover - defensive; NetworkX is in-process
            raise BackendError(
                backend="NetworkX", reason=str(exc), setup_section="networkx"
            ) from exc
        return GraphHealth(
            backend=self.backend_name, version=nx.__version__, nodes=nodes, edges=edges
        )

    def node_count(self) -> int:
        return int(self._graph.number_of_nodes())

    def edge_count(self) -> int:
        return int(self._graph.number_of_edges())

    def nodes(self) -> Iterator[Entity]:
        return iter([self._entity(n) for n in sorted(self._graph.nodes)])

    def edges(self) -> Iterator[Relation]:
        relations: list[Relation] = [r for _, _, r in self._graph.edges(data="relation")]
        return iter(sorted(relations, key=lambda r: r.key))

    @property
    def graph(self) -> Any:
        """The underlying ``nx.MultiDiGraph`` (read-only use, e.g. graph quality metrics)."""
        return self._graph

    # ------------------------------------------------------------------ persistence

    def save(self, path: str | Path) -> Path:
        """Write the graph to ``path`` as JSON, atomically; returns the path."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": GRAPH_FILE_FORMAT,
            "version": GRAPH_FILE_VERSION,
            "nodes": [n.model_dump(mode="json") for n in self.nodes()],
            "edges": [e.model_dump(mode="json") for e in self.edges()],
        }
        fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, sort_keys=True)
            os.replace(tmp, target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return target

    @classmethod
    def load(cls, path: str | Path) -> NetworkxGraphStore:
        """Restore a store written by :meth:`save`; raises ``ValueError`` on a bad file."""
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Graph file {source} is not valid JSON: {exc}") from exc
        if (
            not isinstance(payload, dict)
            or payload.get("format") != GRAPH_FILE_FORMAT
            or payload.get("version") != GRAPH_FILE_VERSION
        ):
            raise ValueError(
                f"Graph file {source} is not an {GRAPH_FILE_FORMAT} v{GRAPH_FILE_VERSION} file"
            )
        store = cls()
        for node in payload.get("nodes", []):
            store.upsert_node(Entity.model_validate(node))
        for edge in payload.get("edges", []):
            store.upsert_edge(Relation.model_validate(edge))
        return store

    # ------------------------------------------------------------------ helpers

    def _entity(self, name: str) -> Entity:
        entity: Entity = self._graph.nodes[name]["entity"]
        return entity

    def _edge(self, source: str, relation: str, target: str) -> Relation | None:
        if not self._graph.has_edge(source, target, key=relation):
            return None
        found: Relation = self._graph.edges[source, target, relation]["relation"]
        return found


__all__ = ["GRAPH_FILE_FORMAT", "GRAPH_FILE_VERSION", "NetworkxGraphStore"]
