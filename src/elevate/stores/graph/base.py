"""Graph_Store protocol, graph data models and the in-memory reference backend (Req 2.2, 2.5).

Every graph backend (NetworkX now; Neo4j, Memgraph and Postgres graph tables in Phase 5)
implements :class:`GraphStore`: node upsert, edge upsert, entity lookup by name, 1-hop and
2-hop neighbour fetch, a ``health_check`` and node/edge counts.

Semantics shared by every backend, so backends are interchangeable (Property 21):

* A node is identified by its ``name`` (surrounding whitespace stripped). Entity resolution
  (merging "Paris" and "PARIS") is the Entity_Resolver's job, not the store's, so names are
  stored exactly. :meth:`GraphStore.entity_by_name` tries the exact name first and then a
  case- and whitespace-insensitive match, picking the lexicographically smallest stored name
  when several match.
* An edge is identified by ``(source, relation, target)``. Edges are directed as stored, but
  neighbour fetches ignore direction, matching LightRAG's undirected knowledge graph.
* Upserts merge with any existing record (:func:`merge_entity`, :func:`merge_relation`). The
  merge keeps every source chunk ID (provenance, Req 15.3) and is idempotent: upserting the
  same record twice leaves the store unchanged.
* Upserting an edge whose endpoint does not exist creates a placeholder node of type
  :data:`UNKNOWN_TYPE` carrying the edge's source chunk IDs.
* Neighbour fetches return a :class:`Subgraph` with the centre plus every node within
  ``hops`` steps, and every edge between two returned nodes, in a deterministic order.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

import elevate

UNKNOWN_TYPE = "UNKNOWN"
"""Type given to placeholder nodes created by an edge upsert."""

EdgeKey = tuple[str, str, str]
"""``(source, relation, target)``: the identity of an edge."""


class _GraphModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _strip_name(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("name must be non-empty")
    return stripped


class Entity(_GraphModel):
    """A graph node: an extracted entity with its provenance (Req 15.1, 15.2)."""

    name: str
    type: str = UNKNOWN_TYPE
    descriptions: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    profile_embedding: list[float] | None = None

    @field_validator("name")
    @classmethod
    def _strip(cls, value: str) -> str:
        return _strip_name(value)


class Relation(_GraphModel):
    """A graph edge ``source -relation-> target`` with its provenance (Req 15.1)."""

    source: str
    relation: str
    target: str
    description: str = ""
    weight: float = Field(default=1.0, ge=0.0)
    source_chunk_ids: list[str] = Field(default_factory=list)

    @field_validator("source", "relation", "target")
    @classmethod
    def _strip(cls, value: str) -> str:
        return _strip_name(value)

    @property
    def key(self) -> EdgeKey:
        return (self.source, self.relation, self.target)


Node = Entity
"""Alias used by the design's ``GraphStore`` protocol."""
Edge = Relation
"""Alias used by the design's ``GraphStore`` protocol."""


class Subgraph(_GraphModel):
    """Result of a neighbour fetch.

    ``nodes`` are ordered by hop distance from the centre, then by name; ``distances`` maps
    each node name to its hop distance (the centre is 0). ``edges`` holds every edge whose
    two endpoints are both in ``nodes``, ordered by ``(source, relation, target)``. An
    unknown centre yields an empty subgraph with ``center=None``.
    """

    center: str | None
    hops: int
    nodes: list[Entity] = Field(default_factory=list)
    edges: list[Relation] = Field(default_factory=list)
    distances: dict[str, int] = Field(default_factory=dict)

    def node_names(self) -> set[str]:
        return {node.name for node in self.nodes}

    def neighbour_names(self) -> set[str]:
        """Names of every returned node except the centre."""
        return {node.name for node in self.nodes if node.name != self.center}


class GraphHealth(_GraphModel):
    """Outcome of a successful health check (Req 2.5)."""

    backend: str
    version: str
    status: Literal["ok"] = "ok"
    nodes: int = 0
    edges: int = 0


@runtime_checkable
class GraphStore(Protocol):
    """The one interface every graph backend implements (Req 2.2)."""

    backend_name: str

    def upsert_node(self, node: Entity) -> None: ...

    def upsert_edge(self, edge: Relation) -> None: ...

    def entity_by_name(self, name: str) -> Entity | None: ...

    def neighbours(self, name: str, hops: int) -> Subgraph: ...

    def health_check(self) -> GraphHealth:
        """Report backend name, version and status; raise ``BackendError`` on failure."""
        ...

    def node_count(self) -> int: ...

    def edge_count(self) -> int: ...

    def nodes(self) -> Iterator[Entity]: ...

    def edges(self) -> Iterator[Relation]: ...


# ------------------------------------------------------------------- shared semantics


def normalise_name(name: str) -> str:
    """Case- and whitespace-insensitive form of a name, used for lenient lookups."""
    return " ".join(name.split()).casefold()


def _union(first: Sequence[str], second: Iterable[str]) -> list[str]:
    """Order-preserving union without duplicates."""
    seen = dict.fromkeys(first)
    seen.update(dict.fromkeys(second))
    return list(seen)


def merge_entity(old: Entity | None, new: Entity) -> Entity:
    """Merge ``new`` into ``old``: union descriptions and chunk IDs, keep a known type.

    The new type wins unless it is :data:`UNKNOWN_TYPE`; a new profile embedding replaces
    the old one, and ``None`` keeps the old one.
    """
    if old is None:
        return new.model_copy(
            update={
                "descriptions": _union([], new.descriptions),
                "source_chunk_ids": _union([], new.source_chunk_ids),
            }
        )
    return Entity(
        name=old.name,
        type=old.type if new.type == UNKNOWN_TYPE else new.type,
        descriptions=_union(old.descriptions, new.descriptions),
        source_chunk_ids=_union(old.source_chunk_ids, new.source_chunk_ids),
        profile_embedding=(
            new.profile_embedding if new.profile_embedding is not None else old.profile_embedding
        ),
    )


def merge_relation(old: Relation | None, new: Relation) -> Relation:
    """Merge ``new`` into ``old``: union description lines and chunk IDs, keep the max weight."""
    if old is None:
        return new.model_copy(update={"source_chunk_ids": _union([], new.source_chunk_ids)})
    lines = _union(old.description.splitlines(), new.description.splitlines())
    return Relation(
        source=old.source,
        relation=old.relation,
        target=old.target,
        description="\n".join(line for line in lines if line),
        weight=max(old.weight, new.weight),
        source_chunk_ids=_union(old.source_chunk_ids, new.source_chunk_ids),
    )


def placeholder_node(name: str, source_chunk_ids: Sequence[str]) -> Entity:
    """Node created for an edge endpoint that has not been upserted yet."""
    return Entity(name=name, type=UNKNOWN_TYPE, source_chunk_ids=list(source_chunk_ids))


def resolve_name(name: str, exact: Mapping[str, object], names: Iterable[str]) -> str | None:
    """Return the stored name matching ``name``: exact first, then normalised."""
    stripped = name.strip()
    if stripped in exact:
        return stripped
    target = normalise_name(stripped)
    if not target:
        return None
    matches = sorted(n for n in names if normalise_name(n) == target)
    return matches[0] if matches else None


def check_hops(hops: int) -> None:
    if hops < 1:
        raise ValueError(f"hops must be >= 1, got {hops}")


def build_subgraph(
    center: str | None,
    hops: int,
    distances: Mapping[str, int],
    node_of: Mapping[str, Entity],
    edges: Iterable[Relation],
) -> Subgraph:
    """Assemble a deterministic :class:`Subgraph` from BFS distances."""
    ordered = sorted(distances, key=lambda n: (distances[n], n))
    inside = set(distances)
    kept = sorted((e for e in edges if e.source in inside and e.target in inside), key=_edge_key)
    return Subgraph(
        center=center,
        hops=hops,
        nodes=[node_of[n] for n in ordered],
        edges=kept,
        distances={n: distances[n] for n in ordered},
    )


def _edge_key(edge: Relation) -> EdgeKey:
    return edge.key


# ------------------------------------------------------------------- reference backend


class InMemoryGraphStore:
    """Dict-adjacency reference implementation of :class:`GraphStore` for tests.

    Pure Python with no third-party graph library, so it doubles as the oracle that other
    backends are compared against (Property 21).
    """

    backend_name = "in-memory"

    def __init__(self) -> None:
        self._nodes: dict[str, Entity] = {}
        self._edges: dict[EdgeKey, Relation] = {}
        # Undirected adjacency: name -> neighbour name -> keys of edges between them.
        self._adj: dict[str, dict[str, set[EdgeKey]]] = {}

    def upsert_node(self, node: Entity) -> None:
        self._nodes[node.name] = merge_entity(self._nodes.get(node.name), node)
        self._adj.setdefault(node.name, {})

    def upsert_edge(self, edge: Relation) -> None:
        for endpoint in (edge.source, edge.target):
            if endpoint not in self._nodes:
                self.upsert_node(placeholder_node(endpoint, edge.source_chunk_ids))
        self._edges[edge.key] = merge_relation(self._edges.get(edge.key), edge)
        self._adj[edge.source].setdefault(edge.target, set()).add(edge.key)
        self._adj[edge.target].setdefault(edge.source, set()).add(edge.key)

    def entity_by_name(self, name: str) -> Entity | None:
        found = resolve_name(name, self._nodes, self._nodes)
        return None if found is None else self._nodes[found]

    def neighbours(self, name: str, hops: int) -> Subgraph:
        check_hops(hops)
        center = resolve_name(name, self._nodes, self._nodes)
        if center is None:
            return Subgraph(center=None, hops=hops)
        distances = {center: 0}
        queue = deque([center])
        while queue:
            current = queue.popleft()
            if distances[current] == hops:
                continue
            for other in self._adj[current]:
                if other not in distances:
                    distances[other] = distances[current] + 1
                    queue.append(other)
        incident = {key for n in distances for keys in self._adj[n].values() for key in keys}
        return build_subgraph(
            center, hops, distances, self._nodes, (self._edges[k] for k in incident)
        )

    def health_check(self) -> GraphHealth:
        return GraphHealth(
            backend=self.backend_name,
            version=elevate.__version__,
            nodes=self.node_count(),
            edges=self.edge_count(),
        )

    def node_count(self) -> int:
        return len(self._nodes)

    def edge_count(self) -> int:
        return len(self._edges)

    def nodes(self) -> Iterator[Entity]:
        return iter([self._nodes[n] for n in sorted(self._nodes)])

    def edges(self) -> Iterator[Relation]:
        return iter([self._edges[k] for k in sorted(self._edges)])


__all__ = [
    "UNKNOWN_TYPE",
    "Edge",
    "EdgeKey",
    "Entity",
    "GraphHealth",
    "GraphStore",
    "InMemoryGraphStore",
    "Node",
    "Relation",
    "Subgraph",
    "build_subgraph",
    "check_hops",
    "merge_entity",
    "merge_relation",
    "normalise_name",
    "placeholder_node",
    "resolve_name",
]
