"""Select and freeze a V4 baseline using only development retrieval evidence."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean
from typing import Any

from .config import Settings
from .data import artifact_lock, dataset_manifest, file_hash, stable_hash, write_json
from .evaluation import passages_by_document, retrieval_metrics
from .experiments import project_code_hash, select_questions
from .pipeline import open_pipeline
from .providers import Cache, Providers


async def tune_v4(
    dataset: Path,
    root: Path,
    out: Path,
    reference: Settings,
    *,
    limit: int | None = None,
    sizes: tuple[int, ...] = (256, 512, 1024),
    depths: tuple[int, ...] = (5, 8, 10),
) -> dict[str, Any]:
    """Keep candidates, fusion, providers and all non-chunk parameters fixed."""
    if (
        not sizes
        or not depths
        or len(set(sizes)) != len(sizes)
        or len(set(depths)) != len(depths)
        or min(sizes) <= reference.overlap
        or min(depths) < 1
        or max(depths) > reference.candidates
    ):
        raise ValueError("V4 tuning requires distinct valid chunk sizes and rerank depths")
    reference.model("reranking")
    manifest = dataset_manifest(dataset)
    questions = select_questions(dataset, "dev", limit, reference.seed)
    eligible = [question for question in questions if question.evidence]
    if not eligible:
        raise ValueError("V4 tuning requires annotated development evidence")
    plan = {
        "dataset": manifest,
        "split": "dev",
        "baseline": asdict(reference),
        "qids": [q.question_id for q in questions],
        "sizes": list(sizes),
        "depths": list(depths),
        "code": project_code_hash(),
    }
    with artifact_lock(out / ".tuning.lock"):
        manifest_path = out / "tuning.json"
        if manifest_path.exists():
            stored = json.loads(manifest_path.read_text())
            if stored["plan"] != plan or file_hash(out / "v4.json") != stored["config_sha256"]:
                raise ValueError(
                    "tuning output is already frozen with a different plan or configuration"
                )
            return stored
        trials: list[dict[str, Any]] = []
        for size in sorted(sizes):
            selected = replace(reference, chunk_size=size, rerank_k=max(depths))
            scores: dict[int, list[dict[str, Any]]] = {depth: [] for depth in depths}
            async with (
                Providers(selected, Cache(root / "cache"), root / "budget.sqlite3") as providers,
                open_pipeline(dataset, root, providers, ["V4"]) as pipeline,
            ):
                by_document = passages_by_document(pipeline.catalog)
                for question in eligible:
                    hits, _ = await pipeline.retrieve(question.text, "V4")
                    ranked = await pipeline.rerank(question.text, "V4", hits)
                    for depth in sorted(depths):
                        subset = ranked[:depth]
                        contexts = [pipeline.catalog[hit.chunk_id] for hit in subset]
                        measurements = retrieval_metrics(
                            question.evidence,
                            subset,
                            pipeline.catalog,
                            contexts,
                            by_document=by_document,
                        )
                        scores[depth].append({"question_id": question.question_id, **measurements})
            for depth in sorted(depths):
                rows = scores[depth]
                trials.append(
                    {
                        "chunk_size": size,
                        "rerank_k": depth,
                        "rows": rows,
                        **{
                            metric: mean(values)
                            if (values := [row[metric] for row in rows if row[metric] is not None])
                            else None
                            for metric in (
                                "gold_context_recall",
                                "gold_context_precision",
                                "ndcg_at_10",
                            )
                        },
                    }
                )
        best = min(
            trials,
            key=lambda item: (
                -item["gold_context_recall"],
                -item["gold_context_precision"],
                -item["ndcg_at_10"] if item["ndcg_at_10"] is not None else float("inf"),
                item["rerank_k"],
                item["chunk_size"],
            ),
        )
        frozen = replace(reference, chunk_size=best["chunk_size"], rerank_k=best["rerank_k"])
        selection = {key: value for key, value in best.items() if key != "rows"}
        write_json(out / "v4.json", asdict(frozen))
        result = {
            "plan": plan,
            "plan_hash": stable_hash(plan),
            "tuned_questions": len(eligible),
            "selected": selection,
            "trials": trials,
            "config_sha256": file_hash(out / "v4.json"),
        }
        write_json(manifest_path, result)
        return result
