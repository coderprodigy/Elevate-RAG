"""Vector database adapters using the same normalized vectors and stable passage IDs."""

import asyncio
import json
import os
import time
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx

from .config import Settings
from .data import Chunk, artifact_lock, write_json
from .providers import Providers, Vector, normalized_vectors
from .retrieval import (
    ExactIndex,
    FaissIndex,
    Hit,
    VectorIndex,
    build_vectors,
    sorted_hits,
    vector_key,
)


def environment(key: str) -> str:
    if not (value := os.environ.get(key)):
        raise ValueError(f"set {key} to use this storage backend")
    return value


class QdrantIndex:
    def __init__(self, name: str, dimensions: int, settings: Settings):
        self.name, self.dimensions = name, dimensions
        self.ef = settings.ef_search
        headers = {"api-key": key} if (key := os.environ.get("QDRANT_API_KEY")) else {}
        self.client = httpx.Client(
            base_url=environment("QDRANT_URL"), headers=headers, timeout=settings.timeout
        )
        try:
            response = self.client.get(f"/collections/{name}")
            if response.status_code == 404:
                self.call(
                    "PUT",
                    "",
                    {
                        "vectors": {"size": dimensions, "distance": "Cosine"},
                        "hnsw_config": {
                            "m": settings.hnsw_m,
                            "ef_construct": settings.ef_construction,
                        },
                    },
                )
                self.call(
                    "PUT", "/index?wait=true", {"field_name": "source", "field_schema": "keyword"}
                )
            else:
                response.raise_for_status()
                params = response.json()["result"]["config"]["params"]["vectors"]
                if params["size"] != dimensions or params["distance"] != "Cosine":
                    raise ValueError(
                        "Qdrant collection is incompatible with the expected embedding"
                    )
        except BaseException:
            self.close()
            raise

    def call(self, method: str, path: str, payload: dict[str, Any]) -> Any:
        response = self.client.request(method, f"/collections/{self.name}{path}", json=payload)
        response.raise_for_status()
        return response.json()["result"]

    def upsert(self, chunks: Sequence[Chunk], vectors: Vector) -> None:
        matrix = normalized_vectors(vectors, len(chunks), self.dimensions)
        for start in range(0, len(chunks), 256):
            points = [
                {
                    "id": str(uuid.uuid5(uuid.NAMESPACE_OID, chunk.chunk_id)),
                    "vector": vector,
                    "payload": {"chunk_id": chunk.chunk_id, "source": chunk.source},
                }
                for chunk, vector in zip(
                    chunks[start : start + 256], matrix[start : start + 256].tolist(), strict=True
                )
            ]
            self.call("PUT", "/points?wait=true", {"points": points})

    def search(
        self, query: Vector, limit: int, *, source: str | None = None, ef: int | None = None
    ) -> list[Hit]:
        payload: dict[str, Any] = {
            "query": normalized_vectors([query], 1, self.dimensions)[0].tolist(),
            "limit": limit,
            "with_payload": ["chunk_id"],
            "params": {"hnsw_ef": ef or self.ef},
        }
        if source is not None:
            payload["filter"] = {"must": [{"key": "source", "match": {"value": source}}]}
        rows = self.call("POST", "/points/query", payload)["points"]
        return sorted_hits(((row["payload"]["chunk_id"], row["score"]) for row in rows), limit)

    def get_vectors(self, ids: Sequence[str]) -> dict[str, list[float]]:
        if not ids:
            return {}
        rows = self.call(
            "POST",
            "/points",
            {
                "ids": [str(uuid.uuid5(uuid.NAMESPACE_OID, key)) for key in ids],
                "with_payload": ["chunk_id"],
                "with_vector": True,
            },
        )
        return {row["payload"]["chunk_id"]: row["vector"] for row in rows}

    def size(self) -> int:
        return int(self.call("POST", "/points/count", {"exact": True})["count"])

    def close(self) -> None:
        self.client.close()


class MilvusIndex:
    def __init__(self, name: str, dimensions: int, settings: Settings):
        from pymilvus import DataType, MilvusClient

        self.name, self.dimensions, self.ef = name, dimensions, settings.ef_search
        self.client = MilvusClient(
            uri=environment("MILVUS_URI"),
            token=os.environ.get("MILVUS_TOKEN", ""),
            db_name=os.environ.get("MILVUS_DB_NAME", "default"),
            timeout=settings.timeout,
        )
        try:
            if not self.client.has_collection(name):
                schema = self.client.create_schema(auto_id=False)
                schema.add_field(
                    field_name="chunk_id",
                    datatype=DataType.VARCHAR,
                    is_primary=True,
                    max_length=128,
                )
                schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=8192)
                schema.add_field(
                    field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimensions
                )
                indexes = self.client.prepare_index_params()
                indexes.add_index(
                    field_name="vector",
                    index_type="HNSW",
                    metric_type="COSINE",
                    params={"M": settings.hnsw_m, "efConstruction": settings.ef_construction},
                )
                indexes.add_index(field_name="source", index_type="INVERTED")
                self.client.create_collection(
                    name, schema=schema, index_params=indexes, consistency_level="Strong"
                )
            self.client.load_collection(name)
        except BaseException:
            self.close()
            raise

    def upsert(self, chunks: Sequence[Chunk], vectors: Vector) -> None:
        matrix = normalized_vectors(vectors, len(chunks), self.dimensions)
        for start in range(0, len(chunks), 256):
            self.client.upsert(
                self.name,
                [
                    {"chunk_id": chunk.chunk_id, "source": chunk.source, "vector": vector}
                    for chunk, vector in zip(
                        chunks[start : start + 256],
                        matrix[start : start + 256].tolist(),
                        strict=True,
                    )
                ],
            )

    def search(
        self, query: Vector, limit: int, *, source: str | None = None, ef: int | None = None
    ) -> list[Hit]:
        filtering = (
            {"filter": "source == {source}", "filter_params": {"source": source}}
            if source is not None
            else {}
        )
        rows = self.client.search(
            collection_name=self.name,
            data=normalized_vectors([query], 1, self.dimensions).tolist(),
            limit=limit,
            anns_field="vector",
            consistency_level="Strong",
            search_params={"metric_type": "COSINE", "params": {"ef": max(limit, ef or self.ef)}},
            **filtering,
        )[0]
        return sorted_hits(((row["chunk_id"], row["distance"]) for row in rows), limit)

    def get_vectors(self, ids: Sequence[str]) -> dict[str, list[float]]:
        if not ids:
            return {}
        rows = self.client.get(
            collection_name=self.name, ids=list(ids), output_fields=["chunk_id", "vector"]
        )
        return {row["chunk_id"]: row["vector"] for row in rows}

    def size(self) -> int:
        [row] = self.client.query(
            collection_name=self.name,
            filter="",
            output_fields=["count(*)"],
            consistency_level="Strong",
        )
        return int(row["count(*)"])

    def close(self) -> None:
        self.client.close()


class PgvectorIndex:
    def __init__(self, name: str, dimensions: int, settings: Settings):
        from pgvector.psycopg import register_vector
        from psycopg import sql
        from psycopg_pool import ConnectionPool

        self.table = sql.Identifier(name)
        self.sql, self.dimensions, self.ef = sql, dimensions, settings.ef_search
        self.pool = ConnectionPool(
            environment("ELEVATE_PGVECTOR_DSN"),
            min_size=1,
            max_size=max(32, settings.concurrency),
            timeout=settings.timeout,
            configure=register_vector,
            open=False,
        )
        try:
            self.pool.open(wait=True)
            with self.pool.connection() as conn:
                conn.execute(
                    sql.SQL(
                        "CREATE TABLE IF NOT EXISTS {} (chunk_id text PRIMARY KEY, source text NOT NULL, vector vector({}))"
                    ).format(self.table, sql.Literal(dimensions))
                )
                conn.execute(
                    sql.SQL(
                        "CREATE INDEX IF NOT EXISTS {} ON {} USING hnsw (vector vector_cosine_ops) WITH (m = {}, ef_construction = {})"
                    ).format(
                        sql.Identifier(name + "_hnsw"),
                        self.table,
                        sql.Literal(settings.hnsw_m),
                        sql.Literal(settings.ef_construction),
                    )
                )
                conn.execute(
                    sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (source)").format(
                        sql.Identifier(name + "_source"),
                        self.table,
                    )
                )
        except BaseException:
            self.close()
            raise

    def upsert(self, chunks: Sequence[Chunk], vectors: Vector) -> None:
        matrix = normalized_vectors(vectors, len(chunks), self.dimensions)
        with self.pool.connection() as conn, conn.cursor() as cursor:
            cursor.executemany(
                self.sql.SQL(
                    "INSERT INTO {} VALUES (%s, %s, %s) ON CONFLICT (chunk_id) DO UPDATE SET source = EXCLUDED.source, vector = EXCLUDED.vector"
                ).format(self.table),
                [
                    (chunk.chunk_id, chunk.source, vector)
                    for chunk, vector in zip(chunks, matrix, strict=True)
                ],
            )

    def search(
        self, query: Vector, limit: int, *, source: str | None = None, ef: int | None = None
    ) -> list[Hit]:
        vector = normalized_vectors([query], 1, self.dimensions)[0]
        where = self.sql.SQL("WHERE source = %s" if source is not None else "")
        params = (vector, source, vector, limit) if source is not None else (vector, vector, limit)
        with self.pool.connection() as conn:
            conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(ef or self.ef),))
            rows = conn.execute(
                self.sql.SQL(
                    "SELECT chunk_id, 1 - (vector <=> %s) FROM {} {} ORDER BY vector <=> %s LIMIT %s"
                ).format(self.table, where),
                params,
            ).fetchall()
        return sorted_hits(rows, limit)

    def get_vectors(self, ids: Sequence[str]) -> dict[str, list[float]]:
        if not ids:
            return {}
        with self.pool.connection() as conn:
            rows = conn.execute(
                self.sql.SQL("SELECT chunk_id, vector FROM {} WHERE chunk_id = ANY(%s)").format(
                    self.table
                ),
                (list(ids),),
            ).fetchall()
        return {key: vector.tolist() for key, vector in rows}

    def size(self) -> int:
        with self.pool.connection() as conn:
            row = conn.execute(
                self.sql.SQL("SELECT count(*) FROM {}").format(self.table)
            ).fetchone()
        if row is None:
            raise RuntimeError("pgvector count returned no row")
        return int(row[0])

    def close(self) -> None:
        self.pool.close()


def open_vector(kind: str, name: str, dimensions: int, settings: Settings) -> VectorIndex:
    if kind == "exact":
        return ExactIndex(dimensions)
    if kind == "faiss":
        return FaissIndex(
            dimensions,
            m=settings.hnsw_m,
            construction=settings.ef_construction,
            search_ef=settings.ef_search,
        )
    adapters: dict[str, Callable[[str, int, Settings], VectorIndex]] = {
        "qdrant": QdrantIndex,
        "milvus": MilvusIndex,
        "pgvector": PgvectorIndex,
    }
    if kind not in adapters:
        raise ValueError(f"{kind}: no independent vector adapter")
    return adapters[kind](name, dimensions, settings)


@asynccontextmanager
async def open_passage_index(
    root: Path,
    chunks_key: str,
    chunks: Sequence[Chunk],
    providers: Providers,
) -> AsyncIterator[VectorIndex]:
    settings = providers.settings
    key = vector_key(settings, chunks_key)
    artifact = root / "vectors" / key
    model = settings.model("embedding")
    begun, offset = time.perf_counter(), len(providers.usage)
    vector = None
    try:
        with artifact_lock(artifact / ".index.lock"):
            vector = open_vector(
                settings.vector_store,
                f"{settings.namespace[:24]}_{key[:20]}",
                model.dimensions,
                settings,
            )
            remote = settings.vector_store not in {"exact", "faiss"}
            faiss_path = artifact / "index.faiss"
            manifest_path = artifact / "index.json"
            contract = {
                "key": key,
                "chunks": len(chunks),
                "vector_store": settings.vector_store,
                "model": asdict(model),
            }
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text())
                if any(manifest.get(field) != value for field, value in contract.items()):
                    raise ValueError(
                        "persisted passage index manifest differs from the corpus contract"
                    )
            if isinstance(vector, FaissIndex) and faiss_path.exists():
                await asyncio.to_thread(vector.restore, faiss_path, chunks)
            elif not remote or await asyncio.to_thread(vector.size) != len(chunks):
                vectors = await build_vectors(providers, chunks, artifact / "vectors.npy")
                await asyncio.to_thread(vector.upsert, chunks, vectors)
                if await asyncio.to_thread(vector.size) != len(chunks):
                    raise RuntimeError("vector backend published a different number of passages")
                if isinstance(vector, FaissIndex):
                    await asyncio.to_thread(vector.save, faiss_path)
            if not manifest_path.exists():
                write_json(
                    manifest_path,
                    {
                        **contract,
                        "seconds": time.perf_counter() - begun,
                        "local_bytes": sum(
                            item.stat().st_size for item in artifact.iterdir() if item.is_file()
                        ),
                        "usage": list(map(asdict, providers.usage[offset:])),
                    },
                )
        yield vector
    finally:
        if vector is not None:
            vector.close()
