"""External reference systems through a predictions contract (FR-X3, D-14).

Microsoft GraphRAG, HippoRAG 2, LlamaIndex PropertyGraphIndex and Neo4j GraphRAG cannot share
LightRAG's index, so they are *not* part of the controlled within-LightRAG comparisons. Run them
with their own tooling on the same prepared corpus/questions, export one JSON object per question:

    {"question_id": "...", "answer": "...",
     "contexts": ["retrieved text", ...],            # optional, enables evidence recall
     "prompt_tokens": 0, "completion_tokens": 0,     # optional
     "latency_s": 0.0, "cost_usd": null}             # optional (null = unknown)

and `ingest_predictions` scores them with exactly the same evaluator and writes a standard run
directory (flagged `external: true`) that reports pick up next to controlled runs.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from elevate_rag.data.prepare import DatasetStore
from elevate_rag.evaluation.harness import score_rows, summarize, write_scored
from elevate_rag.generation.generator import is_refusal
from elevate_rag.providers.llm import BaseLLM
from elevate_rag.utils import DataError, read_jsonl, stable_hash, write_json


def load_predictions(path: Path, known_ids: set[str]) -> list[dict[str, Any]]:
    rows = []
    for r in read_jsonl(path):
        if "question_id" not in r or "answer" not in r:
            raise DataError(f"{path}: every row needs question_id and answer")
        if r["question_id"] not in known_ids:
            raise DataError(f"{path}: unknown question_id {r['question_id']!r}")
        rows.append(r)
    return rows


async def ingest_predictions(
    path: Path,
    *,
    system: str,
    dataset: str,
    data_dir: Path,
    runs_root: Path,
    split: str = "test",
    judge_llm: BaseLLM | None = None,
    judge2_llm: BaseLLM | None = None,
) -> dict[str, Any]:
    ds = DatasetStore.open(data_dir, dataset)
    ds.verify()
    questions = ds.questions(split)
    preds = load_predictions(path, {q.question_id for q in questions})
    by_id = {p["question_id"]: p for p in preds}
    missing = [q.question_id for q in questions if q.question_id not in by_id]
    rows = []
    for q in questions:
        p = by_id.get(q.question_id)
        err = "missing prediction" if p is None else p.get("error")
        ans = "" if p is None else str(p["answer"])
        rows.append(
            {
                "question_id": q.question_id,
                "question": q.question,
                "mode": f"external:{system}",
                "answer": ans,
                "refused": is_refusal(ans) if ans else False,
                "error": err,
                "retrieved_chunk_ids": [],
                "context_item_ids": [],
                "context_items": [],
                "context_texts": list((p or {}).get("contexts") or []),
                "invalid_citations": [],
                "route": {},
                "timings": {"total": float((p or {}).get("latency_s") or 0.0)},
                "prompt_tokens": int((p or {}).get("prompt_tokens") or 0),
                "completion_tokens": int((p or {}).get("completion_tokens") or 0),
                "context_tokens": 0,
                "cost_usd": (p or {}).get("cost_usd"),
            }
        )
    scored = await score_rows(
        rows,
        questions,
        None,
        judge_llm=judge_llm,
        judge2_llm=judge2_llm,
        string_evidence_valid=ds.string_evidence_valid,
    )
    run_id = f"EXT-{system}-{dataset}-{stable_hash([str(path), split], 8)}"
    d = runs_root / "EXT" / run_id
    write_scored(d / "scored.jsonl", scored)
    summary = summarize(scored)
    summary["missing_predictions"] = len(missing)
    write_json(d / "summary.json", summary)
    write_json(
        d / "manifest.json",
        {
            "definition": {
                "run_id": run_id,
                "experiment": "EXT",
                "dataset": dataset,
                "variant": {"name": system, "mode": None},
                "split": split,
                "config_hash": None,
                "config": {"models": {"generator": {"model": f"external:{system}"}}},
            },
            "status": "complete",
            "external": True,
            "demo": False,
            "predictions": str(path),
            "dataset_version": ds.version,
            "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        },
    )
    return summary
