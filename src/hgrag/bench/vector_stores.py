"""Vector store adapters behind one interface, for the E4 Level A micro-benchmark.

Each store gets the same vectors, the same HNSW ``M`` / ``ef_construction`` (where the engine
supports them) and a ``group`` attribute for filtered search. Vectors are L2-normalised, so
inner product is the metric everywhere. Server clients are imported lazily (``uv sync --group
bench``); FAISS runs in-process and is the exact-recall reference.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Protocol

import faiss
import numpy as np

from hgrag.config import HNSWSpec
from hgrag.index.embedders import Vectors

COLLECTION = "hgrag_bench"


class VectorStore(Protocol):
    name: str

    def build(self, vectors: Vectors, groups: Sequence[int]) -> None:
        """Create the collection/index and load all vectors (row number = id)."""
        ...

    def search(self, query: Vectors, k: int, ef: int, group: int | None = None) -> list[int]:
        """Top-``k`` row ids for one query vector (optionally only rows in ``group``)."""
        ...

    def insert(self, row_id: int, vector: Vectors, group: int) -> None: ...

    def drop(self) -> None: ...


class FaissStore:
    """In-process FAISS: ``flat`` (exact) or ``hnsw``."""

    def __init__(self, kind: str, hnsw: HNSWSpec) -> None:
        self.name = f"faiss-{kind}"
        self._kind = kind
        self._hnsw = hnsw
        self._index: Any = None
        self._groups: np.ndarray[Any, Any] = np.zeros(0, dtype=np.int64)

    def build(self, vectors: Vectors, groups: Sequence[int]) -> None:
        dim = vectors.shape[1]
        if self._kind == "flat":
            self._index = faiss.IndexFlatIP(dim)
        else:
            self._index = faiss.IndexHNSWFlat(dim, self._hnsw.m, faiss.METRIC_INNER_PRODUCT)
            self._index.hnsw.efConstruction = self._hnsw.ef_construction
        self._index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self._groups = np.asarray(groups, dtype=np.int64)

    def search(self, query: Vectors, k: int, ef: int, group: int | None = None) -> list[int]:
        params: Any = faiss.SearchParametersHNSW() if self._kind == "hnsw" else None
        if params is not None:
            params.efSearch = ef
        if group is not None:
            # The selector must outlive the search call, so keep a reference on the params object.
            selector = faiss.IDSelectorBatch(np.flatnonzero(self._groups == group).astype(np.int64))
            params = params or faiss.SearchParameters()
            params.sel = selector
            params._selector_ref = selector
        _, rows = self._index.search(query.reshape(1, -1).astype(np.float32), k, params=params)
        return [int(r) for r in rows[0] if r != -1]

    def insert(self, row_id: int, vector: Vectors, group: int) -> None:
        self._index.add(vector.reshape(1, -1).astype(np.float32))
        self._groups = np.append(self._groups, group)

    def drop(self) -> None:
        self._index = None


class QdrantStore:
    name = "qdrant"

    def __init__(self, hnsw: HNSWSpec, url: str | None = None) -> None:
        from qdrant_client import QdrantClient

        self._hnsw = hnsw
        self._client = QdrantClient(
            url=url or os.environ.get("QDRANT_URL", "http://localhost:6333")
        )

    def build(self, vectors: Vectors, groups: Sequence[int]) -> None:
        from qdrant_client import models

        self.drop()
        self._client.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(size=vectors.shape[1], distance=models.Distance.DOT),
            hnsw_config=models.HnswConfigDiff(
                m=self._hnsw.m, ef_construct=self._hnsw.ef_construction
            ),
        )
        self._client.create_payload_index(COLLECTION, "group", models.PayloadSchemaType.INTEGER)
        batch = 1000
        for start in range(0, len(vectors), batch):
            self._client.upsert(
                COLLECTION,
                points=models.Batch(
                    ids=list(range(start, min(start + batch, len(vectors)))),
                    vectors=vectors[start : start + batch].tolist(),
                    payloads=[{"group": int(g)} for g in groups[start : start + batch]],
                ),
                wait=True,
            )

    def search(self, query: Vectors, k: int, ef: int, group: int | None = None) -> list[int]:
        from qdrant_client import models

        flt = (
            models.Filter(
                must=[models.FieldCondition(key="group", match=models.MatchValue(value=group))]
            )
            if group is not None
            else None
        )
        result = self._client.query_points(
            COLLECTION,
            query=query.tolist(),
            limit=k,
            search_params=models.SearchParams(hnsw_ef=ef),
            query_filter=flt,
        )
        return [int(p.id) for p in result.points]

    def insert(self, row_id: int, vector: Vectors, group: int) -> None:
        from qdrant_client import models

        self._client.upsert(
            COLLECTION,
            points=[
                models.PointStruct(id=row_id, vector=vector.tolist(), payload={"group": group})
            ],
            wait=True,
        )

    def drop(self) -> None:
        if self._client.collection_exists(COLLECTION):
            self._client.delete_collection(COLLECTION)


class PgVectorStore:
    name = "pgvector"

    def __init__(self, hnsw: HNSWSpec, dsn: str | None = None) -> None:
        import psycopg

        self._hnsw = hnsw
        self._conn = psycopg.connect(dsn or _postgres_dsn(), autocommit=True)

    def build(self, vectors: Vectors, groups: Sequence[int]) -> None:
        dim = vectors.shape[1]
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"DROP TABLE IF EXISTS {COLLECTION}")
            cur.execute(
                f"CREATE TABLE {COLLECTION} (id int PRIMARY KEY, grp int, emb vector({dim}))"
            )
            with cur.copy(f"COPY {COLLECTION} (id, grp, emb) FROM STDIN") as copy:
                for i, (vector, group) in enumerate(zip(vectors, groups, strict=True)):
                    copy.write_row((i, int(group), _pg_vector(vector)))
            cur.execute(
                f"CREATE INDEX ON {COLLECTION} USING hnsw (emb vector_ip_ops) "
                f"WITH (m = {self._hnsw.m}, ef_construction = {self._hnsw.ef_construction})"
            )
            cur.execute(f"CREATE INDEX ON {COLLECTION} (grp)")

    def search(self, query: Vectors, k: int, ef: int, group: int | None = None) -> list[int]:
        where = "WHERE grp = %s" if group is not None else ""
        sql = f"SELECT id FROM {COLLECTION} {where} ORDER BY emb <#> %s::vector LIMIT %s"
        vector = _pg_vector(query)
        params: tuple[Any, ...] = (group, vector, k) if group is not None else (vector, k)
        with self._conn.cursor() as cur:
            cur.execute(f"SET hnsw.ef_search = {int(ef)}")
            cur.execute(sql, params)
            return [int(row[0]) for row in cur.fetchall()]

    def insert(self, row_id: int, vector: Vectors, group: int) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {COLLECTION} (id, grp, emb) VALUES (%s, %s, %s::vector)",
                (row_id, group, _pg_vector(vector)),
            )

    def drop(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {COLLECTION}")


class MilvusStore:
    name = "milvus"

    def __init__(self, hnsw: HNSWSpec, uri: str | None = None) -> None:
        from pymilvus import MilvusClient

        self._hnsw = hnsw
        self._client = MilvusClient(
            uri=uri or os.environ.get("MILVUS_URI", "http://localhost:19530")
        )

    def build(self, vectors: Vectors, groups: Sequence[int]) -> None:
        from pymilvus import DataType

        self.drop()
        schema = self._client.create_schema(auto_id=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=int(vectors.shape[1]))
        schema.add_field("grp", DataType.INT64)
        index = self._client.prepare_index_params()
        index.add_index(
            field_name="vector",
            index_type="HNSW",
            metric_type="IP",
            params={"M": self._hnsw.m, "efConstruction": self._hnsw.ef_construction},
        )
        self._client.create_collection(COLLECTION, schema=schema, index_params=index)
        rows = [
            {"id": i, "vector": v.tolist(), "grp": int(g)}
            for i, (v, g) in enumerate(zip(vectors, groups, strict=True))
        ]
        for start in range(0, len(rows), 1000):
            self._client.insert(COLLECTION, rows[start : start + 1000])
        self._client.load_collection(COLLECTION)

    def search(self, query: Vectors, k: int, ef: int, group: int | None = None) -> list[int]:
        hits = self._client.search(
            COLLECTION,
            data=[query.tolist()],
            limit=k,
            search_params={"metric_type": "IP", "params": {"ef": max(ef, k)}},
            filter=f"grp == {group}" if group is not None else "",
        )
        return [int(hit["id"]) for hit in hits[0]]

    def insert(self, row_id: int, vector: Vectors, group: int) -> None:
        self._client.insert(COLLECTION, [{"id": row_id, "vector": vector.tolist(), "grp": group}])

    def drop(self) -> None:
        if self._client.has_collection(COLLECTION):
            self._client.drop_collection(COLLECTION)


VECTOR_STORES = ("faiss-flat", "faiss-hnsw", "qdrant", "pgvector", "milvus")


def make_vector_store(name: str, hnsw: HNSWSpec) -> VectorStore:
    if name in ("faiss-flat", "faiss-hnsw"):
        return FaissStore(name.removeprefix("faiss-"), hnsw)
    if name == "qdrant":
        return QdrantStore(hnsw)
    if name == "pgvector":
        return PgVectorStore(hnsw)
    if name == "milvus":
        return MilvusStore(hnsw)
    raise KeyError(f"unknown vector store '{name}'; choose from {VECTOR_STORES}")


def _pg_vector(vector: Vectors) -> str:
    return "[" + ",".join(f"{float(x):.7f}" for x in vector) + "]"


def _postgres_dsn() -> str:
    env = os.environ.get
    return (
        f"host={env('POSTGRES_HOST', 'localhost')} port={env('POSTGRES_PORT', '5432')} "
        f"user={env('POSTGRES_USER', 'hgrag')} "
        f"password={env('POSTGRES_PASSWORD', 'hgrag-dev-password')} "
        f"dbname={env('POSTGRES_DATABASE', 'hgrag')}"
    )
