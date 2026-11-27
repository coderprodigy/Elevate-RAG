"""Run in a framework's own Python environment. Inputs contain corpus and query text only."""

import asyncio
import json
import os
import sys
import time
from collections.abc import Callable
from contextlib import ExitStack
from importlib.metadata import version
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

DIST = {
    "llamaindex-property-graph": "llama-index-core",
    "neo4j-graphrag": "neo4j-graphrag",
    "microsoft-graphrag": "graphrag",
    "hipporag2": "hipporag",
}


def loopback(value: str, schemes: set[str]) -> str:
    url = urlsplit(value)
    if (
        url.scheme not in schemes
        or url.hostname not in {"localhost", "127.0.0.1", "::1"}
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ValueError("framework endpoints must be direct loopback URLs without credentials")
    return value.rstrip("/")


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def checkpoint_ids(path: Path, expected: set[str]) -> set[str]:
    if not path.exists():
        return set()
    finished = set()
    with path.open("r+b") as stream:
        while line := stream.readline():
            if not line.endswith(b"\n"):
                stream.truncate(stream.tell() - len(line))
                stream.flush()
                os.fsync(stream.fileno())
                break
            row = json.loads(line)
            key = row["id"]
            if key not in expected or key in finished or row.get("status") not in {"ok", "error"}:
                raise ValueError("native checkpoint has an invalid question ID or status")
            finished.add(key)
    return finished


def llamaindex(
    docs: list[dict[str, Any]],
    config: dict[str, Any],
    directory: Path,
    stack: ExitStack,
) -> Callable[[str], tuple[str, list[str]]]:
    import httpx
    from llama_index.core import Document, PropertyGraphIndex
    from llama_index.core.base.embeddings.base import BaseEmbedding
    from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
    from llama_index.core.llms import CompletionResponse, CustomLLM, LLMMetadata

    client = stack.enter_context(httpx.Client(timeout=config["request_timeout"], trust_env=False))

    class LocalLLM(CustomLLM):
        @property
        def metadata(self) -> LLMMetadata:
            return LLMMetadata(
                model_name=config["chat_model"],
                context_window=config["context_window"],
                num_output=config["max_output"],
            )

        def complete(
            self, prompt: str, formatted: bool = False, **kwargs: Any
        ) -> CompletionResponse:
            response = client.post(
                config["chat_url"] + "/chat/completions",
                json={
                    "model": config["chat_model"],
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": config["max_output"],
                },
            )
            response.raise_for_status()
            return CompletionResponse(text=response.json()["choices"][0]["message"]["content"])

        def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Any:
            yield self.complete(prompt, formatted=formatted, **kwargs)

    class LocalEmbedding(BaseEmbedding):
        def _get_text_embedding(self, text: str) -> list[float]:
            response = client.post(
                config["embedding_url"] + "/embeddings",
                json={
                    "model": config["embedding_model"],
                    "input": [text],
                },
            )
            response.raise_for_status()
            vector = response.json()["data"][0]["embedding"]
            if len(vector) != config["embedding_dimensions"]:
                raise ValueError("local embedding size differs from the experiment configuration")
            return vector

        def _get_query_embedding(self, query: str) -> list[float]:
            return self._get_text_embedding(query)

        async def _aget_query_embedding(self, query: str) -> list[float]:
            return await asyncio.to_thread(self._get_query_embedding, query)

    llm = LocalLLM()
    embedder = LocalEmbedding(model_name=config["embedding_model"])
    index = PropertyGraphIndex.from_documents(
        [Document(id_=doc["id"], text=doc["text"]) for doc in docs],
        llm=llm,
        embed_model=embedder,
        kg_extractors=[SimpleLLMPathExtractor(llm=llm, num_workers=1)],
        show_progress=False,
    )
    engine = index.as_query_engine(llm=llm, similarity_top_k=config["top_k"], include_text=True)

    def query(text: str) -> tuple[str, list[str]]:
        response = engine.query(text)
        return str(response), [source.node.get_content() for source in response.source_nodes]

    return query


def neo4j_graph(
    docs: list[dict[str, Any]],
    config: dict[str, Any],
    directory: Path,
    stack: ExitStack,
) -> Callable[[str], tuple[str, list[str]]]:
    from neo4j import GraphDatabase
    from neo4j_graphrag.embeddings import OpenAIEmbeddings
    from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
    from neo4j_graphrag.generation import GraphRAG
    from neo4j_graphrag.indexes import create_vector_index
    from neo4j_graphrag.llm import OpenAILLM
    from neo4j_graphrag.retrievers import VectorCypherRetriever

    uri = loopback(config["neo4j_uri"], {"bolt", "bolt+s", "bolt+ssc"})
    database = config["neo4j_database"]
    driver = stack.enter_context(
        GraphDatabase.driver(
            uri,
            auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
        )
    )
    driver.verify_connectivity()
    llm = OpenAILLM(
        config["chat_model"],
        base_url=config["chat_url"],
        api_key="local",
        model_params={"temperature": 0, "max_tokens": config["max_output"]},
    )
    embedder = OpenAIEmbeddings(
        model=config["embedding_model"], api_key="local", base_url=config["embedding_url"]
    )
    index_name = "elevate_reference_passage"
    indexed = directory / "neo4j-index.json"
    if not indexed.exists():
        procedures, _, _ = driver.execute_query(
            "SHOW PROCEDURES YIELD name WHERE name = 'apoc.refactor.mergeNodes' RETURN name",
            database_=database,
        )
        if not procedures:
            raise ValueError("Neo4j framework entity resolution requires the APOC Core plugin")
        records, _, _ = driver.execute_query(
            "MATCH (n) RETURN count(n) AS count", database_=database
        )
        if records[0]["count"]:
            raise ValueError("Neo4j framework comparison requires an empty dedicated database")
        builder = SimpleKGPipeline(
            llm=llm,
            driver=driver,
            embedder=embedder,
            from_file=False,
            neo4j_database=database,
            on_error="RAISE",
        )

        async def index_documents() -> None:
            for doc in docs:
                await builder.run_async(text=doc["text"])

        asyncio.run(index_documents())
        create_vector_index(
            driver,
            index_name,
            "Chunk",
            "embedding",
            config["embedding_dimensions"],
            "cosine",
            neo4j_database=database,
        )
        driver.execute_query("CALL db.awaitIndexes()", database_=database)
        indexed.write_text(json.dumps({"uri": uri, "database": database}))
    elif json.loads(indexed.read_text()) != {"uri": uri, "database": database}:
        raise ValueError("Neo4j framework index belongs to a different database")
    retriever = VectorCypherRetriever(
        driver,
        index_name,
        embedder=embedder,
        neo4j_database=database,
        retrieval_query=(
            "OPTIONAL MATCH (entity)-[:FROM_CHUNK]->(node) "
            "OPTIONAL MATCH (entity)-[rel]-(other) WHERE type(rel) <> 'FROM_CHUNK' "
            "RETURN node.text AS text, score, collect(distinct {entity: entity.name, relation: type(rel), other: other.name}) AS graph"
        ),
    )
    engine = GraphRAG(retriever=retriever, llm=llm)

    def query(text: str) -> tuple[str, list[str]]:
        result = engine.search(
            query_text=text,
            retriever_config={"top_k": config["top_k"]},
            return_context=True,
        )
        return result.answer, [item.content for item in result.retriever_result.items]

    return query


def microsoft(
    docs: list[dict[str, Any]],
    config: dict[str, Any],
    directory: Path,
    stack: ExitStack,
) -> Callable[[str], tuple[str, list[str]]]:
    import pandas as pd
    from graphrag import api
    from graphrag.config.load_config import load_config

    inputs = directory / "input"
    inputs.mkdir(exist_ok=True)
    for index, doc in enumerate(docs):
        (inputs / f"{index:08}.txt").write_text(doc["text"], encoding="utf-8")
    settings = {
        "completion_models": {
            "default_completion_model": {
                "model_provider": "openai",
                "model": config["chat_model"],
                "api_base": config["chat_url"],
                "api_key": "local",
                "call_args": {"temperature": 0},
            }
        },
        "embedding_models": {
            "default_embedding_model": {
                "model_provider": "openai",
                "model": config["embedding_model"],
                "api_base": config["embedding_url"],
                "api_key": "local",
            }
        },
    }
    (directory / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    native_config = load_config(directory)
    marker = directory / "indexed"
    if not marker.exists():
        results = asyncio.run(api.build_index(config=native_config))
        if any(getattr(result, "errors", None) for result in results):
            raise RuntimeError("Microsoft GraphRAG indexing reported workflow errors")
        marker.touch()
    entities = pd.read_parquet(directory / "output" / "entities.parquet")
    communities = pd.read_parquet(directory / "output" / "communities.parquet")
    reports = pd.read_parquet(directory / "output" / "community_reports.parquet")

    def query(text: str) -> tuple[str, list[str]]:
        response, data = asyncio.run(
            api.global_search(
                config=native_config,
                entities=entities,
                communities=communities,
                community_reports=reports,
                community_level=None,
                dynamic_community_selection=False,
                response_type="Concise answer",
                query=text,
            )
        )
        if isinstance(data, dict):
            contexts = [
                frame.to_json(orient="records") if hasattr(frame, "to_json") else json.dumps(frame)
                for frame in data.values()
            ]
        elif isinstance(data, str):
            contexts = [data] if data else []
        else:
            contexts = [frame.to_json(orient="records") for frame in data]
        return response if isinstance(response, str) else json.dumps(response), contexts

    return query


def hipporag(
    docs: list[dict[str, Any]],
    config: dict[str, Any],
    directory: Path,
    stack: ExitStack,
) -> Callable[[str], tuple[str, list[str]]]:
    from hipporag import HippoRAG
    from hipporag.utils.config_utils import BaseConfig

    os.environ["ELEVATE_REFERENCE_LOCAL_KEY"] = "local"
    engine = stack.enter_context(
        HippoRAG(
            global_config=BaseConfig(
                retrieval_top_k=config["top_k"],
                qa_top_k=config["top_k"],
                temperature=0,
                max_new_tokens=config["max_output"],
                embedding_request_timeout=config["request_timeout"],
            ),
            save_dir=str(directory),
            llm_model_name=config["chat_model"],
            llm_base_url=config["chat_url"],
            llm_api_key_env="ELEVATE_REFERENCE_LOCAL_KEY",
            embedding_model_name=config["embedding_model"],
            embedding_provider="openai",
            embedding_base_url=config["embedding_url"],
            embedding_api_key_env="ELEVATE_REFERENCE_LOCAL_KEY",
        )
    )
    engine.index(docs=[doc["text"] for doc in docs])

    def query(text: str) -> tuple[str, list[str]]:
        results = engine.rag_qa(queries=[text])
        solution = results[0][0]
        return solution.answer, list(solution.docs)

    return query


ENGINES = {
    "llamaindex-property-graph": llamaindex,
    "neo4j-graphrag": neo4j_graph,
    "microsoft-graphrag": microsoft,
    "hipporag2": hipporag,
}


def main() -> int:
    export, out, manifest = map(Path, sys.argv[1:])
    run = json.loads(manifest.read_text())
    framework, config = run["framework"], run["settings"]
    if version(DIST[framework]) != config["framework_version"]:
        raise ValueError(f"installed {DIST[framework]} does not match the pinned framework version")
    config["chat_url"] = loopback(config["chat_url"], {"http", "https"})
    config["embedding_url"] = loopback(config["embedding_url"], {"http", "https"})
    queries = read_rows(export / "queries.jsonl")
    raw = out / "native-questions.jsonl"
    expected = {row["id"] for row in queries}
    finished = checkpoint_ids(raw, expected)
    if finished == expected:
        return 0
    directory = out / "framework"
    directory.mkdir(exist_ok=True)
    with ExitStack() as stack, raw.open("a", encoding="utf-8") as stream:
        query = ENGINES[framework](read_rows(export / "documents.jsonl"), config, directory, stack)
        for row in queries:
            if row["id"] in finished:
                continue
            started = time.perf_counter()
            try:
                answer, contexts = query(row["question"])
                if (
                    not isinstance(answer, str)
                    or not answer.strip()
                    or not all(isinstance(item, str) for item in contexts)
                ):
                    raise ValueError("framework returned invalid answer or context")
                output = {"status": "ok", "answer": answer, "contexts": contexts}
            except (RuntimeError, OSError, ValueError, KeyError, TypeError) as exc:
                output = {"status": "error", "error": type(exc).__name__}
            stream.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "latency": time.perf_counter() - started,
                        **output,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
