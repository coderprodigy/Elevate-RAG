"""Vector store adapters (FR-DB1, D-05).

All stores hold L2-normalised vectors and score with inner product (= cosine). HNSW parameters
(`M`, `ef_construction`, `ef_search`) come from :class:`VectorStoreConfig` and are recorded in the
index manifest (FR-V3). `exact` is the dependency-free reference (and the "flat" ground truth for
the Level-A matched-recall protocol).

Server-backed stores read their URL/DSN from the env var named in the config and fail loudly if it
is unreachable — there is no silent fallback (FR-DB7).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from elevate_rag.config import VectorStoreConfig
from elevate_rag.utils import ConfigError, StorageError, guard_openmp

Hit = tuple[str, float]


def _env(cfg: VectorStoreConfig, default_env: str) -> str:
    name = cfg.url_env or default_env
    val = os.getenv(name)
    if not val:
        raise ConfigError(f"vector store {cfg.backend}: environment variable {name} is not set")
    return val


class VectorStore:
    """Minimal interface: add / search / persist. IDs are strings; metadata is a flat dict."""

    backend = "abstract"

    def __init__(self, cfg: VectorStoreConfig, name: str, dim: int) -> None:
        self.cfg = cfg
        self.name = name
        self.dim = dim

    def add(
        self, ids: list[str], vectors: np.ndarray, metadata: list[dict[str, Any]] | None = None
    ) -> None:
        raise NotImplementedError

    def search(self, query: np.ndarray, k: int, flt: dict[str, Any] | None = None) -> list[Hit]:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        """Persist local state (server-backed stores persist themselves)."""

    def set_search_param(self, value: int) -> None:
        """Set the ANN search breadth (`ef` / `nprobe`) — used by the matched-recall sweep."""
        self.cfg = self.cfg.model_copy(update={"hnsw_ef_search": value})

    def close(self) -> None:
        pass

    def _check(self, vectors: np.ndarray) -> np.ndarray:
        v = np.asarray(vectors, dtype=np.float32)
        if v.ndim != 2 or v.shape[1] != self.dim:
            raise StorageError(f"{self.backend}: expected (n, {self.dim}) vectors, got {v.shape}")
        return v


# --------------------------------------------------------------------------- exact (numpy)


class ExactStore(VectorStore):
    backend = "exact"

    def __init__(self, cfg: VectorStoreConfig, name: str, dim: int) -> None:
        super().__init__(cfg, name, dim)
        self._ids: list[str] = []
        self._vecs = np.zeros((0, dim), dtype=np.float32)
        self._meta: list[dict[str, Any]] = []

    def add(
        self, ids: list[str], vectors: np.ndarray, metadata: list[dict[str, Any]] | None = None
    ) -> None:
        v = self._check(vectors)
        if len(ids) != len(v):
            raise StorageError("ids and vectors differ in length")
        self._ids += ids
        self._vecs = np.vstack([self._vecs, v])
        self._meta += metadata or [{} for _ in ids]

    def search(self, query: np.ndarray, k: int, flt: dict[str, Any] | None = None) -> list[Hit]:
        if not self._ids:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        scores = self._vecs @ q
        if flt:
            mask = np.array([all(m.get(k2) == v for k2, v in flt.items()) for m in self._meta])
            scores = np.where(mask, scores, -np.inf)
        k = min(k, len(self._ids))
        idx = np.argpartition(-scores, k - 1)[:k]
        # deterministic: score desc, then id asc
        order = sorted(idx, key=lambda i: (-float(scores[i]), self._ids[i]))
        return [(self._ids[i], float(scores[i])) for i in order if np.isfinite(scores[i])]

    def count(self) -> int:
        return len(self._ids)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / f"{self.name}.npy", self._vecs)
        (path / f"{self.name}.ids.json").write_text(
            json.dumps({"ids": self._ids, "meta": self._meta})
        )

    @classmethod
    def load(cls, cfg: VectorStoreConfig, name: str, path: Path) -> ExactStore:
        vecs = np.load(path / f"{name}.npy")
        meta = json.loads((path / f"{name}.ids.json").read_text())
        store = cls(cfg, name, int(vecs.shape[1]))
        store._vecs, store._ids, store._meta = vecs.astype(np.float32), meta["ids"], meta["meta"]
        return store


# --------------------------------------------------------------------------- FAISS (HNSW)


class FaissStore(VectorStore):
    backend = "faiss"

    def __init__(self, cfg: VectorStoreConfig, name: str, dim: int) -> None:
        super().__init__(cfg, name, dim)
        guard_openmp("faiss")
        try:
            import faiss
        except ImportError as exc:
            raise ConfigError("install the `faiss` extra") from exc
        self._faiss = faiss
        self._index = faiss.IndexHNSWFlat(dim, cfg.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        self._index.hnsw.efConstruction = cfg.hnsw_ef_construction
        self._ids: list[str] = []
        self._meta: list[dict[str, Any]] = []

    def add(
        self, ids: list[str], vectors: np.ndarray, metadata: list[dict[str, Any]] | None = None
    ) -> None:
        self._index.add(self._check(vectors))
        self._ids += ids
        self._meta += metadata or [{} for _ in ids]

    def search(self, query: np.ndarray, k: int, flt: dict[str, Any] | None = None) -> list[Hit]:
        self._index.hnsw.efSearch = max(self.cfg.hnsw_ef_search, k)
        q = np.asarray(query, dtype=np.float32).reshape(1, -1)
        kk = k if not flt else min(len(self._ids), k * 10)
        params = None
        if flt:
            allowed = np.array(
                [i for i, m in enumerate(self._meta) if all(m.get(a) == b for a, b in flt.items())],
                dtype=np.int64,
            )
            params = self._faiss.SearchParametersHNSW(  # type: ignore[call-arg]  # faiss stubs lack kwargs
                sel=self._faiss.IDSelectorArray(allowed), efSearch=self._index.hnsw.efSearch
            )
        scores, idx = self._index.search(q, kk, params=params)
        hits = [(self._ids[i], float(s)) for s, i in zip(scores[0], idx[0], strict=True) if i >= 0]
        return sorted(hits, key=lambda h: (-h[1], h[0]))[:k]

    def count(self) -> int:
        return len(self._ids)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(path / f"{self.name}.faiss"))
        (path / f"{self.name}.ids.json").write_text(
            json.dumps({"ids": self._ids, "meta": self._meta})
        )


# --------------------------------------------------------------------------- Qdrant


class QdrantStore(VectorStore):
    backend = "qdrant"

    def __init__(self, cfg: VectorStoreConfig, name: str, dim: int) -> None:
        super().__init__(cfg, name, dim)
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:
            raise ConfigError("install the `qdrant` extra") from exc
        self._m = models
        self._client = QdrantClient(url=_env(cfg, "ELEVATE_QDRANT_URL"), timeout=60)
        self.collection = f"{cfg.collection_prefix}_{name}"
        try:
            exists = self._client.collection_exists(self.collection)
        except Exception as exc:
            raise StorageError(f"Qdrant unreachable: {exc}") from exc
        if not exists:
            self._client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.DOT),
                hnsw_config=models.HnswConfigDiff(
                    m=cfg.hnsw_m, ef_construct=cfg.hnsw_ef_construction
                ),
            )
            self._client.create_payload_index(
                self.collection, "source", models.PayloadSchemaType.KEYWORD
            )

    def add(
        self, ids: list[str], vectors: np.ndarray, metadata: list[dict[str, Any]] | None = None
    ) -> None:
        v = self._check(vectors)
        meta = metadata or [{} for _ in ids]
        points = [
            self._m.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, i)),
                vector=vec.tolist(),
                payload={"_id": i, **m},
            )
            for i, vec, m in zip(ids, v, meta, strict=True)
        ]
        for s in range(0, len(points), 256):
            self._client.upsert(self.collection, points[s : s + 256], wait=True)

    def search(self, query: np.ndarray, k: int, flt: dict[str, Any] | None = None) -> list[Hit]:
        f = None
        if flt:
            f = self._m.Filter(
                must=[
                    self._m.FieldCondition(key=a, match=self._m.MatchValue(value=b))
                    for a, b in flt.items()
                ]
            )
        res = self._client.query_points(
            self.collection,
            query=np.asarray(query, dtype=np.float32).tolist(),
            limit=k,
            query_filter=f,
            search_params=self._m.SearchParams(hnsw_ef=self.cfg.hnsw_ef_search),
        )
        hits = [(p.payload["_id"], float(p.score)) for p in res.points if p.payload]
        return sorted(hits, key=lambda h: (-h[1], h[0]))

    def count(self) -> int:
        return int(self._client.count(self.collection, exact=True).count)

    def drop(self) -> None:
        self._client.delete_collection(self.collection)

    def close(self) -> None:
        self._client.close()


# --------------------------------------------------------------------------- Milvus


class MilvusStore(VectorStore):
    backend = "milvus"

    def __init__(self, cfg: VectorStoreConfig, name: str, dim: int) -> None:
        super().__init__(cfg, name, dim)
        # Read our URL *before* importing pymilvus: importing it loads ./.env into os.environ
        # (third-party side effect), which would mask a missing configuration.
        uri = _env(cfg, "ELEVATE_MILVUS_URI")
        try:
            from pymilvus import DataType, MilvusClient
        except ImportError as exc:
            raise ConfigError("install the `milvus` extra") from exc
        try:
            self._client = MilvusClient(uri=uri)
        except Exception as exc:  # pymilvus raises raw gRPC errors; normalise (FR-DB7)
            raise StorageError(f"Milvus unreachable at {uri}: {exc}") from exc
        self.collection = f"{cfg.collection_prefix}_{name}"
        if not self._client.has_collection(self.collection):
            schema = self._client.create_schema(auto_id=False, enable_dynamic_field=True)
            schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=512)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
            idx = self._client.prepare_index_params()
            idx.add_index(
                "vector",
                index_type="HNSW",
                metric_type="IP",
                params={"M": cfg.hnsw_m, "efConstruction": cfg.hnsw_ef_construction},
            )
            self._client.create_collection(self.collection, schema=schema, index_params=idx)
        self._client.load_collection(self.collection)

    def add(
        self, ids: list[str], vectors: np.ndarray, metadata: list[dict[str, Any]] | None = None
    ) -> None:
        v = self._check(vectors)
        meta = metadata or [{} for _ in ids]
        rows = [
            {"id": i, "vector": vec.tolist(), **m} for i, vec, m in zip(ids, v, meta, strict=True)
        ]
        for s in range(0, len(rows), 512):
            self._client.upsert(self.collection, rows[s : s + 512])
        self._client.flush(self.collection)

    def search(self, query: np.ndarray, k: int, flt: dict[str, Any] | None = None) -> list[Hit]:
        expr = " and ".join(f'{a} == "{b}"' for a, b in (flt or {}).items()) or ""
        res = self._client.search(
            self.collection,
            [np.asarray(query).tolist()],
            limit=k,
            filter=expr,
            search_params={"metric_type": "IP", "params": {"ef": max(k, self.cfg.hnsw_ef_search)}},
        )
        hits = [(r["id"], float(r["distance"])) for r in res[0]]
        return sorted(hits, key=lambda h: (-h[1], h[0]))

    def count(self) -> int:
        return int(self._client.query(self.collection, output_fields=["count(*)"])[0]["count(*)"])

    def drop(self) -> None:
        self._client.drop_collection(self.collection)

    def close(self) -> None:
        self._client.close()


# --------------------------------------------------------------------------- pgvector


class PgVectorStore(VectorStore):
    backend = "pgvector"

    def __init__(self, cfg: VectorStoreConfig, name: str, dim: int) -> None:
        super().__init__(cfg, name, dim)
        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except ImportError as exc:
            raise ConfigError("install the `postgres` extra") from exc
        try:
            self._conn = psycopg.connect(_env(cfg, "ELEVATE_PG_DSN"), autocommit=True)
        except psycopg.OperationalError as exc:
            raise StorageError(f"PostgreSQL unreachable: {exc}") from exc
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self._conn)
        self.table = f"{cfg.collection_prefix}_{name}".replace("-", "_")
        self._conn.execute(
            f"CREATE TABLE IF NOT EXISTS {self.table} (id text PRIMARY KEY, "
            f"embedding vector({dim}), meta jsonb)"
        )
        self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS {self.table}_hnsw ON {self.table} USING hnsw "
            f"(embedding vector_ip_ops) WITH (m = {cfg.hnsw_m}, "
            f"ef_construction = {cfg.hnsw_ef_construction})"
        )

    def add(
        self, ids: list[str], vectors: np.ndarray, metadata: list[dict[str, Any]] | None = None
    ) -> None:
        v = self._check(vectors)
        meta = metadata or [{} for _ in ids]
        with self._conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {self.table} (id, embedding, meta) VALUES (%s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding, "
                "meta = EXCLUDED.meta",
                [(i, vec, json.dumps(m)) for i, vec, m in zip(ids, v, meta, strict=True)],
            )

    def search(self, query: np.ndarray, k: int, flt: dict[str, Any] | None = None) -> list[Hit]:
        self._conn.execute(f"SET hnsw.ef_search = {max(k, self.cfg.hnsw_ef_search)}")
        where = ""
        params: list[Any] = [np.asarray(query, dtype=np.float32)]
        if flt:
            where = "WHERE meta @> %s::jsonb"
            params.append(json.dumps(flt))
        rows = self._conn.execute(
            f"SELECT id, -(embedding <#> %s) AS score FROM {self.table} {where} "
            f"ORDER BY embedding <#> %s LIMIT {int(k)}",
            [params[0], *params[1:], params[0]],
        ).fetchall()
        return sorted(((r[0], float(r[1])) for r in rows), key=lambda h: (-h[1], h[0]))

    def count(self) -> int:
        row = self._conn.execute(f"SELECT count(*) FROM {self.table}").fetchone()
        if row is None:
            raise StorageError(f"count(*) on {self.table} returned no row")
        return int(row[0])

    def drop(self) -> None:
        self._conn.execute(f"DROP TABLE IF EXISTS {self.table}")

    def close(self) -> None:
        self._conn.close()


_STORES: dict[str, type[VectorStore]] = {
    "exact": ExactStore,
    "faiss": FaissStore,
    "qdrant": QdrantStore,
    "milvus": MilvusStore,
    "pgvector": PgVectorStore,
}


def make_vector_store(cfg: VectorStoreConfig, name: str, dim: int) -> VectorStore:
    return _STORES[cfg.backend](cfg, name, dim)


def wait_until_searchable(
    store: VectorStore, probe_id: str, probe: np.ndarray, timeout_s: float = 30.0
) -> float:
    """Write-to-searchable delay (FR-DB4): seconds until a freshly added vector is returned."""
    t0 = time.perf_counter()
    store.add([probe_id], probe.reshape(1, -1), [{"source": "_probe"}])
    while time.perf_counter() - t0 < timeout_s:
        if any(h[0] == probe_id for h in store.search(probe, 5)):
            return time.perf_counter() - t0
        time.sleep(0.005)
    raise StorageError(f"{store.backend}: probe not searchable within {timeout_s}s")
