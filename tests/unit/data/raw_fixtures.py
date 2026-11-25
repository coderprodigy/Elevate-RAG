"""Writers for tiny raw files in each source's real format (checked against the downloads)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from hgrag.config import DatasetSettings
from hgrag.data.loaders.base import BuildInputs


def make_inputs(tmp: Path, name: str, raw_name: str | None = None, **settings: Any) -> BuildInputs:
    raw = tmp / "data" / "raw" / (raw_name or name)
    processed = tmp / "data" / "processed"
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    return BuildInputs(
        name=name,
        raw_dir=raw,
        processed_root=processed,
        settings=DatasetSettings(source_url="test", license="test", **settings),
        seed=750,
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_json_lines(path: Path, values: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(v) + "\n" for v in values), encoding="utf-8")


def _ctx(title: str, text: str, pid: str) -> dict[str, str]:
    return {"title": title, "text": text, "passage_id": pid}


def write_nq(raw_dir: Path) -> None:
    records = [
        {
            "dataset": "nq_dev",
            "question": f"question {i}",
            "answers": [f"answer {i}"],
            "positive_ctxs": [_ctx(f"Gold {i}", f"gold text {i}", f"g{i}")],
            "negative_ctxs": [],
            "hard_negative_ctxs": [
                _ctx(f"Neg {i}.{j}", f"neg text {i}.{j}", f"n{i}{j}") for j in range(3)
            ],
        }
        for i in range(3)
    ]
    with gzip.open(raw_dir / "biencoder-nq-dev.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(records, handle)


def write_hotpotqa(raw_dir: Path) -> None:
    rows = [
        {
            "id": "h1",
            "question": "Which is older, A or B?",
            "answer": "A",
            "type": "comparison",
            "level": "hard",
            "supporting_facts": {"title": ["A", "B"], "sent_id": [0, 0]},
            "context": {
                "title": ["A", "B", "C"],
                "sentences": [
                    ["A was built in 1900.", " It is old."],
                    ["B was built in 1950."],
                    ["C."],
                ],
            },
        },
        {
            "id": "h2",
            "question": "Where was the founder of A born?",
            "answer": "Tallis",
            "type": "bridge",
            "level": "hard",
            "supporting_facts": {"title": ["A", "D"], "sent_id": [0, 0]},
            "context": {
                "title": ["A", "D", "C"],
                "sentences": [
                    ["A was built in 1900.", " It is old."],
                    ["D was born in Tallis."],
                    ["C."],
                ],
            },
        },
    ]
    path = raw_dir / "distractor" / "validation-00000-of-00001.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def musique_record(
    qid: str, answerable: bool, paragraphs: list[tuple[str, bool]], answer: str = "ans"
) -> dict[str, Any]:
    return {
        "id": qid,
        "question": f"question {qid}",
        "question_decomposition": [],
        "answer": answer,
        "answer_aliases": ["alias"],
        "answerable": answerable,
        "paragraphs": [
            {"idx": i, "title": f"T {text}", "paragraph_text": text, "is_supporting": supporting}
            for i, (text, supporting) in enumerate(paragraphs)
        ],
    }


def write_multihop_rag(raw_dir: Path) -> None:
    corpus = [
        {
            "title": f"Article {i}",
            "author": None,
            "source": "Wire",
            "published_at": f"2023-10-0{i + 1}T00:00:00+00:00",
            "category": "tech",
            "url": f"https://example.com/{i}",
            "body": f"Body of article {i}.\n\nSecond paragraph {i}.",
        }
        for i in range(2)
    ]
    evidence = [
        {"title": "Article 0", "fact": "fact zero", "source": "Wire"},
        {"title": "Article 1", "fact": "fact one", "source": "Wire"},
    ]
    queries = [
        {
            "query": "Who did X?",
            "answer": "Y",
            "question_type": "inference_query",
            "evidence_list": evidence,
        },
        {
            "query": "Who did Z?",
            "answer": "Insufficient information.",
            "question_type": "null_query",
            "evidence_list": [],
        },
    ]
    write_json(raw_dir / "corpus.json", corpus)
    write_json(raw_dir / "MultiHopRAG.json", queries)


def write_graphrag_bench(raw_dir: Path) -> None:
    write_json(
        raw_dir / "Datasets/Corpus/novel.json",
        [
            {"corpus_name": "Novel-1", "context": "Once upon a time."},
            {"corpus_name": "Novel-2", "context": "The end."},
        ],
    )
    questions = [
        {
            "id": f"Novel-{i}",
            "source": "Novel-1",
            "question": f"q{i}?",
            "answer": "a",
            "question_type": t,
            "evidence": ["e"],
        }
        for i, t in enumerate(
            ["Fact Retrieval", "Complex Reasoning", "Contextual Summarize", "Creative Generation"]
        )
    ]
    questions.append(
        {
            "id": "Novel-x",
            "source": "Novel-404",
            "question": "orphan?",
            "answer": "a",
            "question_type": "Fact Retrieval",
            "evidence": [],
        }
    )
    write_json(raw_dir / "Datasets/Questions/novel_questions.json", questions)
