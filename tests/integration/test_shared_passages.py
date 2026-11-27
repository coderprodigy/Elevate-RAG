import asyncio

import numpy as np
import pytest

from elevate_rag.data import Chunk
from elevate_rag.retrieval import ExactIndex


def test_shared_view_uses_one_vector_and_validates_graph_writes():
    pytest.importorskip("lightrag")
    from lightrag.utils import EmbeddingFunc

    from elevate_rag.graph_passages import SharedPassages, native_passage_id

    chunks = [
        Chunk("first", "d1", "Repeated passage", "Identical text", "source", 0, 14),
        Chunk("repeat", "d1", "Repeated passage", "Identical text", "source", 20, 34),
        Chunk("other", "d2", "Unique passage", "Unique text", "source", 0, 11),
    ]
    index = ExactIndex(2)
    index.upsert(chunks, np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.float32))

    async def embed(texts, context):
        assert context == "query"
        return np.asarray([[1, 0]] * len(texts), dtype=np.float32)

    view = SharedPassages(
        index, chunks, "test_workspace", EmbeddingFunc(2, embed, supports_asymmetric=True)
    )
    original = native_passage_id(chunks[0])
    other = native_passage_id(chunks[2])

    async def check():
        assert [row["id"] for row in await view.query("query", 2)] == [original, other]
        assert [row["id"] for row in await view.get_by_ids([original, "missing"])] == [original]
        assert (await view.get_by_ids_strict([original, "missing"])) == [
            await view.get_by_id(original),
            None,
        ]
        assert await view.get_vectors_by_ids([original, other, "missing"]) == {
            original: [1.0, 0.0],
            other: [0.0, 1.0],
        }
        assert not await view.is_empty()
        assert await view.get_exact_count() == 2
        await view.upsert({original: {"content": "Identical text"}})
        with pytest.raises(ValueError, match="change shared"):
            await view.upsert({original: {"content": "Altered"}})
        await view.delete([original])
        assert index.size() == 3
        assert (await view.drop())["status"] == "error"
        with pytest.raises(NotImplementedError):
            await view.delete_entity("fake")
        with pytest.raises(NotImplementedError):
            await view.delete_entity_relation("fake")
        assert await view.query("query", 0) == []

    asyncio.run(check())
