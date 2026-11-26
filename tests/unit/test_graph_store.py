from __future__ import annotations

import json

import networkx as nx
import pytest
from pydantic import ValidationError

from elevate.config.schema import BackendSettings
from elevate.core.errors import BackendError
from elevate.stores.graph import (
    Entity,
    GraphStore,
    InMemoryGraphStore,
    NetworkxGraphStore,
    Relation,
    create_graph_store,
)
from elevate.stores.graph.base import UNKNOWN_TYPE

BACKENDS = [InMemoryGraphStore, NetworkxGraphStore]


@pytest.fixture(params=BACKENDS, ids=lambda cls: cls.backend_name)
def store(request) -> GraphStore:
    return request.param()


def build_chain(store: GraphStore) -> None:
    """A -> B -> C -> D plus E isolated; B <- X (reverse direction)."""
    for name in "ABCDEX":
        store.upsert_node(Entity(name=name, type="T", source_chunk_ids=[f"chunk-{name}"]))
    store.upsert_edge(Relation(source="A", relation="r", target="B", source_chunk_ids=["c1"]))
    store.upsert_edge(Relation(source="B", relation="r", target="C", source_chunk_ids=["c2"]))
    store.upsert_edge(Relation(source="C", relation="r", target="D", source_chunk_ids=["c3"]))
    store.upsert_edge(Relation(source="X", relation="r", target="B", source_chunk_ids=["c4"]))


def test_both_backends_satisfy_protocol(store):
    assert isinstance(store, GraphStore)


def test_node_upsert_merges_provenance_and_is_idempotent(store):
    first = Entity(name="Paris", type="CITY", descriptions=["capital"], source_chunk_ids=["c1"])
    store.upsert_node(first)
    store.upsert_node(first)
    store.upsert_node(
        Entity(name="Paris", type=UNKNOWN_TYPE, descriptions=["in France"], source_chunk_ids=["c2"])
    )
    got = store.entity_by_name("Paris")
    assert got is not None
    assert got.type == "CITY"
    assert got.descriptions == ["capital", "in France"]
    assert got.source_chunk_ids == ["c1", "c2"]
    assert store.node_count() == 1


def test_edge_upsert_merges_and_creates_placeholder_endpoints(store):
    edge = Relation(source="A", relation="knows", target="B", weight=0.5, source_chunk_ids=["c1"])
    store.upsert_edge(edge)
    store.upsert_edge(edge)
    store.upsert_edge(
        Relation(
            source="A",
            relation="knows",
            target="B",
            description="d2",
            weight=2.0,
            source_chunk_ids=["c2"],
        )
    )
    assert store.edge_count() == 1
    assert store.node_count() == 2
    (merged,) = list(store.edges())
    assert merged.weight == 2.0
    assert merged.description == "d2"
    assert merged.source_chunk_ids == ["c1", "c2"]
    placeholder = store.entity_by_name("B")
    assert placeholder is not None
    assert placeholder.type == UNKNOWN_TYPE
    assert placeholder.source_chunk_ids == ["c1"]


def test_distinct_relations_between_same_pair_are_kept(store):
    store.upsert_edge(Relation(source="A", relation="r1", target="B"))
    store.upsert_edge(Relation(source="A", relation="r2", target="B"))
    assert store.edge_count() == 2


def test_entity_lookup_exact_then_case_insensitive(store):
    store.upsert_node(Entity(name="New York"))
    store.upsert_node(Entity(name="NEW YORK"))
    assert store.entity_by_name("New York").name == "New York"
    assert store.entity_by_name("  new   york ").name == "NEW YORK"
    assert store.entity_by_name("Boston") is None
    assert store.entity_by_name("") is None


def test_one_and_two_hop_neighbours_ignore_direction(store):
    build_chain(store)
    one = store.neighbours("B", hops=1)
    assert one.center == "B"
    assert one.neighbour_names() == {"A", "C", "X"}
    assert one.distances["B"] == 0
    two = store.neighbours("A", hops=2)
    assert two.neighbour_names() == {"B", "C", "X"}
    assert two.distances == {"A": 0, "B": 1, "C": 2, "X": 2}
    assert [n.name for n in two.nodes] == ["A", "B", "C", "X"]
    assert {e.key for e in two.edges} == {("A", "r", "B"), ("B", "r", "C"), ("X", "r", "B")}
    assert store.neighbours("E", hops=2).neighbour_names() == set()


def test_neighbours_unknown_name_and_bad_hops(store):
    build_chain(store)
    empty = store.neighbours("missing", hops=1)
    assert empty.center is None and empty.nodes == [] and empty.edges == []
    with pytest.raises(ValueError):
        store.neighbours("A", hops=0)


def test_backends_agree_on_neighbourhoods():
    reference, nx_store = InMemoryGraphStore(), NetworkxGraphStore()
    for s in (reference, nx_store):
        build_chain(s)
    for name in "ABCDEX":
        for hops in (1, 2):
            assert reference.neighbours(name, hops) == nx_store.neighbours(name, hops)
    assert list(reference.nodes()) == list(nx_store.nodes())
    assert list(reference.edges()) == list(nx_store.edges())


def test_health_check_reports_backend_version_and_counts(store):
    build_chain(store)
    health = store.health_check()
    assert health.backend == store.backend_name
    assert health.status == "ok"
    assert (health.nodes, health.edges) == (6, 4)
    if isinstance(store, NetworkxGraphStore):
        assert health.version == nx.__version__


def test_models_reject_blank_names():
    with pytest.raises(ValidationError):
        Entity(name="   ")
    with pytest.raises(ValidationError):
        Relation(source="A", relation="", target="B")
    assert Entity(name="  Paris ").name == "Paris"


def test_networkx_save_load_round_trip(tmp_path):
    original = NetworkxGraphStore()
    build_chain(original)
    original.upsert_node(Entity(name="A", profile_embedding=[0.1, 0.2]))
    path = original.save(tmp_path / "sub" / "graph.json")
    restored = NetworkxGraphStore.load(path)
    assert list(restored.nodes()) == list(original.nodes())
    assert list(restored.edges()) == list(original.edges())
    assert restored.neighbours("A", 2) == original.neighbours("A", 2)
    assert not list(tmp_path.joinpath("sub").glob("*.tmp"))


def test_networkx_load_rejects_foreign_files(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"format": "other"}))
    with pytest.raises(ValueError):
        NetworkxGraphStore.load(bad)
    bad.write_text("{not json")
    with pytest.raises(ValueError):
        NetworkxGraphStore.load(bad)


def test_factory_defaults_to_networkx_and_loads_existing_file(tmp_path):
    assert isinstance(create_graph_store(), NetworkxGraphStore)
    assert isinstance(create_graph_store(BackendSettings()), NetworkxGraphStore)
    saved = NetworkxGraphStore()
    build_chain(saved)
    path = saved.save(tmp_path / "g.json")
    loaded = create_graph_store("networkx", path=path)
    assert loaded.node_count() == 6
    assert create_graph_store("networkx", path=tmp_path / "absent.json").node_count() == 0


@pytest.mark.parametrize("backend", ["neo4j", "memgraph", "postgres"])
def test_factory_database_backends_raise_backend_error(backend):
    with pytest.raises(BackendError) as info:
        create_graph_store(BackendSettings(graph=backend))
    assert f"docs/SETUP.md#{backend}" in str(info.value)
