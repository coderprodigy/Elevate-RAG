"""Dataset registry: where each dataset comes from and which loader builds it."""

from __future__ import annotations

from dataclasses import dataclass

from hgrag.data.loaders.base import Builder
from hgrag.data.loaders.graphrag_bench import (
    build_graphrag_bench_medical,
    build_graphrag_bench_novel,
)
from hgrag.data.loaders.hotpotqa import build_hotpotqa
from hgrag.data.loaders.multihop_rag import build_multihop_rag
from hgrag.data.loaders.musique import build_musique
from hgrag.data.loaders.nq import build_nq
from hgrag.data.loaders.ragrouter_bench import build_ragrouter_bench


@dataclass(frozen=True)
class HFFile:
    """A file inside a Hugging Face dataset repository."""

    repo: str
    path: str


@dataclass(frozen=True)
class URLFile:
    url: str
    path: str  # where to save it, relative to the dataset's raw directory


Source = HFFile | URLFile


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    raw_name: str  # folder under data/raw (GraphRAG-Bench subsets share one)
    sources: tuple[Source, ...]
    builder: Builder


_GRB_REPO = "GraphRAG-Bench/GraphRAG-Bench"
_GRB_FILES = tuple(
    HFFile(_GRB_REPO, f"Datasets/{kind}")
    for kind in (
        "Corpus/novel.json",
        "Corpus/medical.json",
        "Questions/novel_questions.json",
        "Questions/medical_questions.json",
    )
)

# Build order matters: ragrouter_bench deduplicates against everything built before it.
REGISTRY: dict[str, DatasetSpec] = {
    spec.name: spec
    for spec in (
        DatasetSpec(
            "nq",
            "nq",
            (
                URLFile(
                    "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz",
                    "biencoder-nq-dev.json.gz",
                ),
            ),
            build_nq,
        ),
        DatasetSpec(
            "hotpotqa",
            "hotpotqa",
            (HFFile("hotpotqa/hotpot_qa", "distractor/validation-00000-of-00001.parquet"),),
            build_hotpotqa,
        ),
        DatasetSpec(
            "musique",
            "musique",
            (HFFile("bdsaglam/musique", "musique_full_v1.0_dev.jsonl"),),
            build_musique,
        ),
        DatasetSpec(
            "multihop_rag",
            "multihop_rag",
            (
                HFFile("yixuantt/MultiHopRAG", "corpus.json"),
                HFFile("yixuantt/MultiHopRAG", "MultiHopRAG.json"),
            ),
            build_multihop_rag,
        ),
        DatasetSpec(
            "graphrag_bench_novel", "graphrag_bench", _GRB_FILES, build_graphrag_bench_novel
        ),
        DatasetSpec(
            "graphrag_bench_medical", "graphrag_bench", _GRB_FILES, build_graphrag_bench_medical
        ),
        DatasetSpec(
            "ragrouter_bench",
            "ragrouter_bench",
            tuple(
                HFFile("Chaplain0908/RAGRouter", f"data/{domain}/Question.json")
                for domain in ("musique", "quality", "ultraDomain_legal", "graphragBench_medical")
            ),
            build_ragrouter_bench,
        ),
    )
}


def resolve(names: list[str]) -> list[DatasetSpec]:
    """Specs for ``names`` ("all" = every dataset) in registry (build) order."""
    if names == ["all"]:
        return list(REGISTRY.values())
    unknown = sorted(set(names) - set(REGISTRY))
    if unknown:
        raise KeyError(f"unknown dataset(s) {unknown}; choose from {list(REGISTRY)}")
    return [spec for name, spec in REGISTRY.items() if name in names]
