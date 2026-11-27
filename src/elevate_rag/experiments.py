"""Experiment design, immutable manifests and resumable per-question execution."""

import itertools
import json
import platform
import re
import sqlite3
import sys
import uuid
from collections.abc import Sequence
from contextlib import closing
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .config import GRAPH_MODES, MODES, VECTOR_MODES, Settings
from .data import (
    Question,
    artifact_lock,
    dataset_manifest,
    file_hash,
    questions_at,
    stable_hash,
    write_json,
    write_jsonl,
)
from .datasets import MAIN
from .evaluation import audit_samples, grade, passages_by_document
from .pipeline import open_pipeline
from .providers import BudgetExceeded, Cache, Providers, Usage
from .routing import SVM, cheapest_correct, classification

STATIC = ("naive", "local", "global", "hybrid", "mix", "V4")


def design(block: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    def add(dataset: str, mode: str, factor: str, variant: Any, **options: Any) -> None:
        cases.append(
            {
                "block": block,
                "dataset": dataset,
                "mode": mode,
                "factor": factor,
                "variant": variant,
                **options,
            }
        )

    if block == "E1":
        for dataset in MAIN:
            for mode in VECTOR_MODES:
                add(dataset, mode, "retrieval", mode)
    elif block == "E2":
        for dataset in MAIN:
            for mode in GRAPH_MODES:
                add(dataset, mode, "retrieval", mode)
        for dataset in ("multihop-rag", "graphrag-novel"):
            for framework in (
                "llamaindex-property-graph",
                "neo4j-graphrag",
                "microsoft-graphrag",
                "hipporag2",
            ):
                add(dataset, "mix", "external-framework", framework, isolated=True)
    elif block == "E3":
        for dataset in MAIN:
            for mode in STATIC:
                add(dataset, mode, "router", "R0")
            for router in ("R1", "R2-query-type", "R2-outcome", "R3", "R4", "offline-oracle"):
                add(dataset, "mix", "router", router)
        add("ragrouter-bench", "mix", "router", "R2-query-type", folds=5)
    elif block == "E4":
        for dataset in ("multihop-rag", "hotpotqa"):
            for store in (
                "qdrant",
                "milvus",
                "pgvector",
                "faiss",
                "neo4j",
                "memgraph",
                "postgres",
                "networkx",
            ):
                add(
                    dataset,
                    "mix",
                    "storage",
                    store,
                    level="B",
                    questions=300,
                    warmup=1000,
                    repetitions=3,
                )
        for store in ("qdrant", "milvus", "pgvector", "faiss"):
            add(
                "multihop-rag",
                "mix",
                "vector-storage",
                store,
                level="A",
                concurrency=[1, 8, 32],
                target_recall=0.95,
            )
        for store in ("neo4j", "memgraph", "postgres", "networkx"):
            add(
                "multihop-rag",
                "local",
                "graph-storage",
                store,
                level="A",
                entities=1000,
                hops=[1, 2],
            )
    elif block == "E5":
        levels = {
            "embedding": ("small-open", "strong-open", "large-open", "api"),
            "reranking": ("none", "open", "api"),
            "extraction": ("small-local", "cheap-api", "strong-api"),
            "generation": ("small-open", "mid-open", "strong-api"),
        }
        for dataset in MAIN:
            for role, tiers in levels.items():
                if role == "extraction" and dataset not in {"multihop-rag", "graphrag-medical"}:
                    continue
                for tier in tiers:
                    add(dataset, "mix", role, tier)
    elif block == "E6":
        for dataset in ("hotpotqa", "musique", "graphrag-novel", "graphrag-medical"):
            for mode in ("naive", "local", "mix"):
                for tier in ("small-open", "mid-open", "strong-api"):
                    add(dataset, mode, "generation-interaction", tier)
    elif block == "ablations":
        factors: dict[str, Sequence[Any]] = {
            "chunk_size": (256, 512, 1024),
            "graph_merge_threshold": (0.85, 0.92, 0.98),
            "context_budget": (4096, 8192, 16384),
            "graph_depth": (1, 2),
            "community": (False, True),
        }
        for factor, values in factors.items():
            for value in values:
                add("multihop-rag", "mix", factor, value)
        for dataset in ("natural-questions", "hotpotqa", "musique"):
            add(dataset, "V4", "paragraph_chunks", True)
    else:
        raise ValueError("experiment block must be E1 through E6 or ablations")
    return cases


def required_modes(mode: str, router: str, settings: Settings, svm: SVM | None) -> list[str]:
    if mode not in MODES:
        raise ValueError(f"unknown retrieval mode {mode}")
    if router == "R0":
        return [mode]
    if router == "R4":
        return [settings.initial_route, "mix"]
    if router == "R2" and svm is not None and svm.strategy == "outcome":
        return sorted(svm.labels)
    if router == "R2" and svm is None:
        raise ValueError("R2 requires a classifier")
    if router in {"R1", "R2", "R3"}:
        return sorted({"V4", "local", "mix", "global", *settings.rules})
    raise ValueError(f"unknown router {router}")


def usage_summary(events: Sequence[Usage]) -> dict[str, Any]:
    charged = [event for event in events if not event.cached]
    by_role = list(map(asdict, events))

    def summed(field: str, *, account_for_cache: bool = True) -> int | float | None:
        values = [getattr(event, field) for event in (charged if account_for_cache else events)]
        return sum(values) if None not in values else None

    prompt = summed("prompt_tokens")
    completion = summed("completion_tokens")
    token_total = prompt + completion if prompt is not None and completion is not None else None
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "tokens": token_total,
        "estimated_usd": summed("estimated_usd"),
        "has_unknown_paid_usage": any(event.estimated_usd is None for event in charged),
        "events": by_role,
    }


def select_questions(dataset: Path, split: str, limit: int | None, seed: int) -> list[Question]:
    partitions = ("dev", "heldout") if split == "all" else (split,)
    available = [question for part in partitions for question in questions_at(dataset, part)]
    ordered = sorted(available, key=lambda question: stable_hash(seed, question.question_id))
    chosen = ordered[:limit]
    if not chosen or (limit is not None and (limit < 1 or len(chosen) != limit)):
        raise ValueError("requested question subset is empty or exceeds the available split")
    return chosen


def dependency_versions() -> dict[str, str]:
    output = {}
    for package in (
        "elevate-rag",
        "numpy",
        "lightrag-hku",
        "networkx",
        "sentence-transformers",
        "scikit-learn",
        "faiss-cpu",
        "qdrant-client",
        "pymilvus",
        "psycopg",
        "neo4j",
        "pgvector",
        "ragas",
        "pyarrow",
        "igraph",
        "matplotlib",
        "llama-index-core",
        "langfuse",
        "openai",
        "httpx",
    ):
        try:
            output[package] = version(package)
        except PackageNotFoundError:
            continue
    return output


def plan_run(
    dataset: Path,
    settings: Settings,
    *,
    mode: str,
    router: str,
    split: str,
    limit: int | None,
    uncached: bool,
    svm: SVM | None = None,
    warmup: int = 0,
) -> tuple[dict[str, Any], list[Question]]:
    if warmup < 0:
        raise ValueError("warm-up count must be nonnegative")
    manifest = dataset_manifest(dataset)
    if settings.synthetic != manifest["synthetic"]:
        raise ValueError("configuration and dataset must agree on synthetic versus real execution")
    if split == "all" and not settings.synthetic:
        raise ValueError("real evaluation must select dev or heldout explicitly")
    if (
        router == "R2"
        and not settings.synthetic
        and (
            svm is None
            or svm.training_dataset != manifest["dataset"]
            or svm.training_dev_sha256 != manifest["files"]["dev.jsonl"]
        )
    ):
        raise ValueError(
            "real R2 experiments require a classifier trained on this dataset's current development split"
        )
    questions = select_questions(dataset, split, limit, settings.seed)
    required = required_modes(mode, router, settings, svm)
    for role in (
        "generation",
        *(("embedding",) if any(arm not in {"V0", "V1"} for arm in required) else ()),
    ):
        settings.model(role)
    if set(required).intersection(set(GRAPH_MODES) - {"naive"}):
        settings.model("extraction")
        settings.model("keywords")
    if any(arm == "V4" or arm in GRAPH_MODES for arm in required):
        if "reranking" not in settings.models:
            raise ValueError("configure a reranker explicitly for tuned or matched retrieval modes")
    if router == "R3":
        settings.model("router")
    generation = settings.model("generation")
    estimate = (
        generation.context_window * generation.input_per_million
        + generation.max_output * generation.output_per_million
    )
    return {
        "schema": 1,
        "dataset": manifest,
        "split": split,
        "qids": [q.question_id for q in questions],
        "mode": mode,
        "router": router,
        "required_modes": required,
        "svm": stable_hash(asdict(svm)) if svm else None,
        "configuration": asdict(settings),
        "configuration_hash": settings.fingerprint,
        "uncached": uncached,
        "warmup_questions": warmup,
        "generation_only_max_usd": estimate
        * (len(questions) + warmup)
        * (2 if router == "R4" else 1)
        / 1e6,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": dependency_versions(),
        "code_hash": project_code_hash(),
    }, questions


def project_code_hash() -> str:
    return stable_hash(
        {path.name: file_hash(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
    )


def initialize_run(db: sqlite3.Connection, payload: dict[str, Any], out: Path) -> None:
    db.execute(
        "CREATE TABLE IF NOT EXISTS manifest (id INTEGER PRIMARY KEY CHECK (id = 1), data TEXT NOT NULL)"
    )
    db.execute(
        "CREATE TABLE IF NOT EXISTS outcomes (id TEXT PRIMARY KEY, status TEXT NOT NULL, data TEXT NOT NULL)"
    )
    serialized = json.dumps(payload, sort_keys=True)
    with db:
        db.execute("INSERT OR IGNORE INTO manifest (id, data) VALUES (1, ?)", (serialized,))
        row = db.execute("SELECT data FROM manifest WHERE id = 1").fetchone()
        if row[0] != serialized:
            raise ValueError("run ID already exists with a different immutable manifest")
    file = out / "manifest.json"
    if file.exists() and json.loads(file.read_text()) != payload:
        raise ValueError("manifest export differs from the immutable run manifest")
    if not file.exists():
        write_json(file, payload)


def results_rows(db: sqlite3.Connection, ordered: Sequence[Question]) -> list[dict[str, Any]]:
    payloads = {row[0]: json.loads(row[1]) for row in db.execute("SELECT id, data FROM outcomes")}
    return [payloads[q.question_id] for q in ordered if q.question_id in payloads]


async def run(
    dataset: Path,
    root: Path,
    results: Path,
    settings: Settings,
    *,
    mode: str = "V4",
    router: str = "R0",
    split: str = "heldout",
    limit: int | None = None,
    run_id: str | None = None,
    retry_failed: bool = False,
    uncached: bool = False,
    svm: SVM | None = None,
    warmup: int = 0,
) -> Path:
    payload, questions = plan_run(
        dataset,
        settings,
        mode=mode,
        router=router,
        split=split,
        limit=limit,
        uncached=uncached,
        svm=svm,
        warmup=warmup,
    )
    key = run_id or f"{mode.lower()}-{uuid.uuid4().hex[:16]}"
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", key):
        raise ValueError("invalid run ID")
    out = results / ("synthetic" if settings.synthetic else "research") / key
    out.mkdir(parents=True, exist_ok=True)
    from .reporting import report_run

    with (
        artifact_lock(out / ".run.lock"),
        closing(sqlite3.connect(out / "checkpoint.sqlite3", timeout=60)) as db,
    ):
        initialize_run(db, {"run_id": key, **payload}, out)
        done = dict(db.execute("SELECT id, status FROM outcomes"))
        ready = [
            q
            for q in questions
            if q.question_id not in done or (retry_failed and done[q.question_id] == "error")
        ]
        halted = None
        try:
            if ready:
                async with (
                    Providers(
                        settings, Cache(root / "cache"), root / "budget.sqlite3", uncached=uncached
                    ) as providers,
                    open_pipeline(
                        dataset,
                        root,
                        providers,
                        payload["required_modes"],
                        require_graph_quality=True,
                    ) as pipeline,
                ):
                    by_document = passages_by_document(pipeline.catalog)
                    if warmup:
                        warmup_started = len(providers.usage)
                        warmup_questions = questions_at(dataset, "dev")
                        if not warmup_questions:
                            raise ValueError("warm-up requires a populated development split")
                        for question in itertools.islice(itertools.cycle(warmup_questions), warmup):
                            await pipeline.answer(question.text, mode=mode, level=router, svm=svm)
                        write_json(
                            out / "warmup.json",
                            {
                                "queries": warmup,
                                "partition": "dev",
                                **usage_summary(providers.usage[warmup_started:]),
                            },
                        )
                    for question in ready:
                        qid = question.question_id
                        base = {
                            "run_id": key,
                            "question_id": qid,
                            "dataset": payload["dataset"]["dataset"],
                            "dataset_version": payload["dataset"]["version"],
                            "mode": mode,
                            "router": router,
                            "query_type": question.query_type,
                            "domain": question.domain,
                            "settings_hash": settings.fingerprint,
                            "time": datetime.now(UTC).isoformat(),
                        }
                        offset = len(providers.usage)
                        try:
                            with providers.tracing.observe(
                                "experiment-question",
                                {"run_id": key, "question_id": qid, "dataset": base["dataset"]},
                            ) as observation:
                                answer = await pipeline.answer(
                                    question.text, mode=mode, level=router, svm=svm
                                )
                                judge_start = len(providers.usage)
                                metrics = await grade(
                                    question,
                                    answer,
                                    pipeline.catalog,
                                    providers,
                                    by_document=by_document,
                                    judge=bool(
                                        settings.models.get("judge")
                                        and settings.models["judge"].provider != "disabled"
                                    ),
                                )
                                if observation is not None:
                                    observation.update(
                                        metadata={
                                            "run_id": key,
                                            "question_id": qid,
                                            "dataset": base["dataset"],
                                            "route": answer.route.final,
                                            "correct": metrics["correct"],
                                        }
                                    )
                            row = {
                                **base,
                                "status": "ok",
                                "answer": asdict(answer.answer),
                                "route": asdict(answer.route),
                                "candidates": list(map(asdict, answer.candidates)),
                                "ranked": list(map(asdict, answer.ranked)),
                                "context": list(map(asdict, answer.context.passages)),
                                "context_tokens": answer.context.context_tokens,
                                "latency": answer.seconds,
                                "trace": answer.telemetry,
                                "metrics": metrics,
                                "correct": metrics["correct"],
                                "evaluation_usage": usage_summary(providers.usage[judge_start:]),
                                **usage_summary(answer.usage),
                            }
                        except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
                            # HTTP payloads and provider configuration can contain credentials.
                            row = {
                                **base,
                                "status": "error",
                                "correct": False,
                                "error": type(exc).__name__,
                                "metrics": {"correct": False},
                                **usage_summary(providers.usage[offset:]),
                            }
                            if isinstance(exc, BudgetExceeded):
                                halted = exc
                        with db:
                            db.execute(
                                "INSERT OR REPLACE INTO outcomes VALUES (?, ?, ?)",
                                (qid, row["status"], json.dumps(row, allow_nan=False)),
                            )
                        if halted:
                            break
        finally:
            rows = results_rows(db, questions)
            write_jsonl(out / "questions.jsonl", rows)
            report_run(out, rows, expected=len(questions), seed=settings.seed)
            audit_samples(
                rows, {question.question_id: question for question in questions}, settings.seed, out
            )
        if halted:
            raise halted
    return out


def router_metrics(
    routed: Sequence[dict[str, Any]],
    modes: Sequence[dict[str, Any]],
    mix: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    by_question = {row["question_id"]: row for row in routed}
    mix_rows = {row["question_id"]: row for row in mix}
    ids = set(by_question)
    if not ids or ids != mix_rows.keys() or ids != {row["question_id"] for row in modes}:
        raise ValueError("router, mix and oracle outcomes must use aligned questions")
    groups: dict[str, set[str]] = {}
    for row in modes:
        arm = row["mode"]
        qid = row["question_id"]
        if qid in groups.setdefault(arm, set()):
            raise ValueError(f"duplicate oracle outcome for {arm}")
        groups[arm].add(qid)
    if any(group != ids for group in groups.values()):
        raise ValueError("oracle modes must cover every aligned question")
    oracle = cheapest_correct(modes)
    all_known = all(row.get("correct") is not None for row in modes)
    correct = (
        sum(row.get("correct") is True for row in routed) / len(ids)
        if all(row.get("correct") is not None for row in routed)
        else None
    )
    oracle_ids = {row["question_id"] for row in modes if row.get("correct") is True}
    oracle_accuracy = len(oracle_ids) / len(ids) if all_known else None
    known = sorted(set(oracle) & {key for key, row in by_question.items() if row.get("route")})

    def savings(field: str) -> float | None:
        base = [row[field] for row in mix if row.get(field) is not None]
        used = [row[field] for row in routed if row.get(field) is not None]
        if len(base) != len(mix) or len(used) != len(routed) or not sum(base):
            return None
        return 1 - sum(used) / sum(base)

    return {
        **classification(
            [oracle[key]["mode"] for key in known],
            [by_question[key]["route"]["final"] for key in known],
        ),
        "routing_label_count": len(known),
        "answer_accuracy": correct,
        "oracle_accuracy": oracle_accuracy,
        "oracle_regret": oracle_accuracy - correct
        if oracle_accuracy is not None and correct is not None
        else None,
        "fallback_fraction": sum(
            bool(row.get("route", {}).get("fallback_reason")) for row in routed
        )
        / len(ids),
        "token_savings_vs_mix": savings("tokens"),
        "cost_savings_vs_mix": savings("estimated_usd"),
    }
