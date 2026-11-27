"""Command-line interface for local research workflows."""

import argparse
import asyncio
import json
import platform
import sys
from collections.abc import Sequence
from contextlib import asynccontextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import httpx

from .benchmarks import bulk_load_graph, graph_microbenchmark, vector_microbenchmark
from .config import MODES, Settings
from .data import dataset_manifest, file_hash, questions_at, read_jsonl, write_json, write_jsonl
from .datasets import DATASETS, PUBLIC_FILES, download, fixture, prepare_dataset
from .evaluation import audit_agreement, judge_pair
from .experiments import (
    dependency_versions,
    design,
    plan_run,
    required_modes,
    router_metrics,
    run,
    select_questions,
)
from .graph import build_graph, graph_key, inspect_graph, open_graph
from .interop import (
    FRAMEWORKS,
    attach_graphrag_metrics,
    export_graphrag_eval,
    export_reference,
    grade_reference,
    import_reference,
    run_reference,
)
from .pipeline import index_parts, open_pipeline
from .providers import Cache, Providers
from .reporting import compare_runs, plot_model_study, plot_runs, plot_vector_curves, report_run
from .retrieval import build_vectors, vector_key
from .routing import (
    SVM,
    calibrate_threshold,
    cheapest_correct,
    cross_validate,
    route_for_query_type,
)
from .statistics import minimum_detectable_effect
from .storage import open_passage_index
from .suite import execute_suite, prepare_arms
from .tuning import tune_v4


def configure() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="elevate-rag", description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic.toml"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name: str, description: str) -> argparse.ArgumentParser:
        return commands.add_parser(name, help=description)

    command("environment", "Show interpreter, platform and installed dependency versions")
    alias = command("prepare-fixture", "Normalize the synthetic fixture")
    alias.add_argument("out", type=Path)
    alias.add_argument("--fixtures", type=Path, default=Path("tests/fixtures"))

    fetch = command("download", "Cache public dataset input files")
    fetch.add_argument("dataset", choices=DATASETS)
    fetch.add_argument("out", type=Path)
    fetch.add_argument(
        "--sha256-file", type=Path, help="JSON mapping filenames to expected SHA-256"
    )

    prepare = command("prepare", "Normalize a real public dataset")
    prepare.add_argument("dataset", choices=DATASETS)
    prepare.add_argument("out", type=Path)
    prepare.add_argument("inputs", type=Path, nargs="+")
    prepare.add_argument("--version", required=True)
    prepare.add_argument("--seed", type=int, default=42)

    validate = command("validate-data", "Check normalized dataset file hashes")
    validate.add_argument("dataset", type=Path)

    index = command("index", "Build a vector or graph index")
    index.add_argument("dataset", type=Path)
    index.add_argument("--graph", action="store_true")
    index.add_argument("--pilot", action="store_true", help="Build a separate 5% graph sample")

    graph = command(
        "inspect-graph", "Audit LightRAG graph shape, provenance and development coverage"
    )
    graph.add_argument("dataset", type=Path)
    graph.add_argument("out", type=Path)

    validate_index = command("validate-index", "Reload and validate a vector or graph index")
    validate_index.add_argument("dataset", type=Path)
    validate_index.add_argument("--mode", choices=MODES, default="V4")

    tune = command("tune-v4", "Select a V4 configuration using only the development split")
    tune.add_argument("dataset", type=Path)
    tune.add_argument("out", type=Path)
    tune.add_argument("--limit", type=int)
    tune.add_argument("--chunk-sizes", nargs="+", type=int, default=[256, 512, 1024])
    tune.add_argument("--rerank-depths", nargs="+", type=int, default=[5, 8, 10])
    answer = command("answer", "Produce a cited answer for a single question")
    answer.add_argument("dataset", type=Path)
    answer.add_argument("question")
    with_modes(answer)

    retrieve = command("retrieve", "Run retrieval, optionally followed by reranking")
    retrieve.add_argument("dataset", type=Path)
    retrieve.add_argument("question")
    retrieve.add_argument("--mode", choices=MODES, default="V4")
    retrieve.add_argument("--rerank", action="store_true")

    experiment = command("run", "Execute or resume one experiment arm")
    experiment.add_argument("dataset", type=Path)
    with_modes(experiment)
    experiment.add_argument("--split", choices=("dev", "heldout", "all"), default="heldout")
    experiment.add_argument("--limit", type=int)
    experiment.add_argument("--run-id")
    experiment.add_argument("--retry-failed", action="store_true")
    experiment.add_argument("--uncached", action="store_true")
    experiment.add_argument("--dry-run", action="store_true")

    suite = command("suite", "Validate or execute a block using frozen model tiers and datasets")
    suite.add_argument("block", choices=("E1", "E2", "E3", "E4", "E5", "E6", "ablations"))
    suite.add_argument("spec", type=Path)
    suite.add_argument("--split", choices=("dev", "heldout"), default="heldout")
    suite.add_argument("--execute", action="store_true")
    suite.add_argument("--retry-failed", action="store_true")
    suite.add_argument("--output", type=Path)
    matrix = command("matrix", "Enumerate the controlled experiment design")
    matrix.add_argument("block", choices=("E1", "E2", "E3", "E4", "E5", "E6", "ablations"))

    train = command("train-router", "Train a classifier using development questions only")
    train.add_argument("dataset", type=Path)
    train.add_argument("out", type=Path)
    train.add_argument("--outcome-run", type=Path, action="append", default=[])
    train.add_argument("--cross-validate", action="store_true")
    train.add_argument("--leave-domain-out", action="store_true")

    router_cv = command(
        "router-cv", "Evaluate router labels with deterministic out-of-fold predictions"
    )
    router_cv.add_argument("dataset", type=Path)
    router_cv.add_argument("--leave-domain-out", action="store_true")
    router_cv.add_argument("--additional-dev", type=Path, action="append", default=[])
    calibrate = command("calibrate-router", "Tune R4 confidence using paired development arms")
    calibrate.add_argument("initial", type=Path)
    calibrate.add_argument("mix", type=Path)

    bench = command("benchmark-vector", "Run Level A vector benchmarks against FAISS exact")
    bench.add_argument("dataset", type=Path)
    bench.add_argument("out", type=Path)
    bench.add_argument(
        "--backend", required=True, choices=("exact", "faiss", "qdrant", "milvus", "pgvector")
    )
    bench.add_argument("--ef", type=int, nargs="+", default=[32, 64, 100, 200])
    bench.add_argument("--limit", type=int, default=100)
    bench.add_argument("--warmup", type=int, default=1000)
    bench.add_argument("--repeats", type=int, default=3)

    graph_bench = command(
        "benchmark-graph", "Benchmark entity lookup, traversal and local retrieval"
    )
    graph_bench.add_argument("dataset", type=Path)
    graph_bench.add_argument("out", type=Path)
    graph_bench.add_argument("--entities", type=int, default=1000)
    graph_bench.add_argument("--limit", type=int, default=100)
    graph_bench.add_argument("--bulk-load", action="store_true")

    evaluate = command("evaluate", "Regenerate run-level metrics from saved outcomes")
    evaluate.add_argument("run", type=Path)
    compare = command("compare", "Perform paired tests and Holm correction")
    compare.add_argument("runs", type=Path, nargs="+")
    powr = command("power", "Estimate paired minimum detectable effect from two development arms")
    powr.add_argument("left", type=Path)
    powr.add_argument("right", type=Path)
    powr.add_argument("sample_size", type=int)

    pair = command("judge-pair", "Compare aligned saved answers in both presentation orders")
    pair.add_argument("dataset", type=Path)
    pair.add_argument("left", type=Path)
    pair.add_argument("right", type=Path)
    pair.add_argument("out", type=Path)

    router_report = command(
        "evaluate-router", "Evaluate router regret and savings against static outcomes"
    )
    router_report.add_argument("routed", type=Path)
    router_report.add_argument("mix", type=Path)
    router_report.add_argument("static", type=Path, nargs="+")
    audit = command("audit-agreement", "Compute agreement from actual human labels")
    audit.add_argument("annotations", type=Path)
    plot = command("report", "Generate measured comparison plots from saved runs")
    plot.add_argument("out", type=Path)
    plot.add_argument("runs", type=Path, nargs="+")
    curve = command("report-vector-benchmark", "Plot measured ANN recall versus query throughput")
    curve.add_argument("csv", type=Path)
    curve.add_argument("out", type=Path)
    model_plot = command("report-model-study", "Plot an executed E5 or E6 suite")
    model_plot.add_argument("suite", type=Path)
    model_plot.add_argument("out", type=Path)
    outgoing = command("export-reference", "Prepare an isolated framework corpus and query set")
    outgoing.add_argument("dataset", type=Path)
    outgoing.add_argument("out", type=Path)
    outgoing.add_argument("--framework", choices=FRAMEWORKS, required=True)
    outgoing.add_argument("--split", choices=("dev", "heldout"), default="heldout")
    outgoing.add_argument("--limit", type=int)
    outgoing.add_argument("--seed", type=int, default=42)
    native = command(
        "run-reference", "Execute an exported study in an isolated native framework environment"
    )
    native.add_argument("dataset", type=Path)
    native.add_argument("exported", type=Path)
    native.add_argument("out", type=Path)
    native.add_argument("--python", type=Path, required=True)
    native.add_argument("--settings", type=Path, required=True)
    native.add_argument("--timeout", type=float, default=3600)
    incoming = command(
        "import-reference", "Align externally measured outcomes and produce an isolated report"
    )
    incoming.add_argument("dataset", type=Path)
    incoming.add_argument("exported", type=Path)
    incoming.add_argument("outcomes", type=Path)
    incoming.add_argument("out", type=Path)
    incoming.add_argument("--version", required=True)
    grade_native = command(
        "grade-reference",
        "Grade an aligned native framework run with the configured independent judge",
    )
    grade_native.add_argument("dataset", type=Path)
    grade_native.add_argument("run", type=Path)
    grade_native.add_argument("out", type=Path)
    upstream = command(
        "export-graphrag-eval", "Export saved outcomes for official GraphRAG-Bench evaluation"
    )
    upstream.add_argument("dataset", type=Path)
    upstream.add_argument("run", type=Path)
    upstream.add_argument("out", type=Path)
    metrics = command(
        "attach-graphrag-metrics", "Join official detailed scores to an evaluation copy"
    )
    metrics.add_argument("run", type=Path)
    metrics.add_argument("scores", type=Path)
    metrics.add_argument("out", type=Path)
    return parser


def with_modes(command: argparse.ArgumentParser) -> None:
    command.add_argument("--mode", choices=MODES, default="V4")
    command.add_argument("--router", choices=("R0", "R1", "R2", "R3", "R4"), default="R0")
    command.add_argument("--svm", type=Path)


def read_run(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((path / "manifest.json").read_text())
    rows = list(read_jsonl(path / "questions.jsonl"))
    if len(rows) != len(manifest["qids"]) or {row["question_id"] for row in rows} != set(
        manifest["qids"]
    ):
        raise ValueError(f"{path}: run is incomplete or has misaligned outcomes")
    return manifest, rows


def train_router(args: argparse.Namespace) -> dict[str, Any]:
    manifest = dataset_manifest(args.dataset)
    dev = questions_at(args.dataset, "dev")
    labels = None
    if args.outcome_run:
        candidates = []
        expected = {question.question_id for question in dev}
        shapes = set()
        for path in args.outcome_run:
            run_manifest, rows = read_run(path)
            row_ids = {row["question_id"] for row in rows}
            shapes.add(frozenset(row_ids))
            if (
                run_manifest["split"] != "dev"
                or run_manifest["dataset"] != manifest
                or not row_ids <= expected
            ):
                raise ValueError(
                    "outcome labels require aligned runs from this dataset's development split"
                )
            if run_manifest.get("router") != "R0":
                raise ValueError("outcome labels require fixed retrieval modes")
            candidates.extend(rows)
        if len(shapes) != 1:
            raise ValueError("all outcome runs must use the same development questions")
        labels = {key: row["mode"] for key, row in cheapest_correct(candidates).items()}
    examples = [
        (
            question.question_id,
            question.text,
            question.query_type if labels is None else labels[question.question_id],
        )
        for question in dev
        if labels is None or question.question_id in labels
    ]
    model = replace(
        SVM.train(examples, "query-type" if labels is None else "outcome"),
        training_dataset=manifest["dataset"],
        training_dev_sha256=manifest["files"]["dev.jsonl"],
    )
    model.save(args.out)
    result: dict[str, Any] = {
        "rows": len(examples),
        "label_strategy": model.strategy,
        "class_labels": model.labels,
        "constant_class": len(model.labels) == 1,
        "training_digest": model.training_digest,
    }
    if args.cross_validate or args.leave_domain_out:
        result["folds"] = cross_validate(
            dev, args.leave_domain_out, manifest["seed"], outcome_labels=labels
        )
    return result


@asynccontextmanager
async def existing_graph(dataset: Path, root: Path, providers: Providers):
    catalog, shared_key, _ = index_parts(dataset, root, providers)
    key = graph_key(providers.settings, shared_key)
    path = root / "graphs" / key / "full"
    if not (path / "index.json").is_file():
        raise ValueError("graph index is missing or incomplete")
    chunks = list(catalog.values())
    async with (
        open_passage_index(root, shared_key, chunks, providers) as vector,
        open_graph(
            path, key, chunks, providers, providers.tokenizer(context=False), vector=vector
        ) as graph,
    ):
        yield graph, catalog


async def operation(args: argparse.Namespace, settings: Settings) -> Any:
    name = args.command
    root, results = args.artifacts, args.results
    svm = SVM.read(args.svm) if getattr(args, "svm", None) else None
    if name == "tune-v4":
        tuning = await tune_v4(
            args.dataset,
            root,
            args.out,
            settings,
            limit=args.limit,
            sizes=tuple(args.chunk_sizes),
            depths=tuple(args.rerank_depths),
        )
        return {
            "config": str(args.out / "v4.json"),
            "tuned_questions": tuning["tuned_questions"],
            "selected": tuning["selected"],
        }
    if name == "suite":
        arms, excluded = prepare_arms(args.spec, args.block, settings)
        outcomes = await execute_suite(
            arms,
            split=args.split,
            root=root,
            results=results,
            dry_run=not args.execute,
            retry_failed=args.retry_failed,
        )
        payload = {
            "block": args.block,
            "dry_run": not args.execute,
            "arms": outcomes,
            "separate_or_pending": excluded,
        }
        if args.output:
            write_json(args.output, payload)
        return payload
    if name == "run":
        options = dict(
            mode=args.mode,
            router=args.router,
            split=args.split,
            limit=args.limit,
            uncached=args.uncached,
            svm=svm,
        )
        if args.dry_run:
            payload, _ = plan_run(args.dataset, settings, **options)
            return payload
        path = await run(
            args.dataset,
            root,
            results,
            settings,
            **options,
            run_id=args.run_id,
            retry_failed=args.retry_failed,
        )
        summary = json.loads((path / "summary.json").read_text())
        return {
            "run": str(path),
            "complete": summary["complete"],
            "attempted": summary["attempted"],
            "errors": summary["groups"][0]["errors"],
        }

    async with Providers(
        settings,
        Cache(root / "cache"),
        root / "budget.sqlite3",
        uncached=name.startswith("benchmark"),
    ) as providers:
        if name == "grade-reference":
            return await grade_reference(args.dataset, args.run, args.out, providers)
        if name == "index":
            if args.pilot and not args.graph:
                raise ValueError("--pilot is only supported with --graph")
            if args.graph:
                catalog, shared_key, _ = index_parts(args.dataset, root, providers)
                key = graph_key(settings, shared_key)
                return await build_graph(
                    root / "graphs" / key / ("pilot" if args.pilot else "full"),
                    key,
                    list(catalog.values()),
                    providers,
                    providers.tokenizer(context=False),
                    pilot=args.pilot,
                    root=root,
                    chunks_key=shared_key,
                )
            async with open_pipeline(args.dataset, root, providers, ["V2"]) as pipeline:
                return {"chunks": len(pipeline.catalog), "behavior_hash": pipeline.index_key}
        if name in {"inspect-graph", "benchmark-graph"}:
            async with existing_graph(args.dataset, root, providers) as (graph, catalog):
                if name == "inspect-graph":
                    nodes, edges = await graph.snapshot()
                    stats = inspect_graph(
                        nodes,
                        edges,
                        questions_at(args.dataset, "dev"),
                        settings.seed,
                        args.out,
                        catalog=catalog,
                    )
                    indexed = json.loads((graph.path / "index.json").read_text())
                    expanded = {
                        **stats,
                        "graph_key": indexed["key"],
                        "indexing": indexed,
                        "development_sha256": file_hash(args.dataset / "dev.jsonl"),
                    }
                    write_json(graph.path / "quality.json", expanded)
                    write_json(args.out / "graph-quality.json", expanded)
                    return expanded
                qs = select_questions(args.dataset, "dev", args.limit, settings.seed)
                measurement = await graph_microbenchmark(
                    graph, [q.text for q in qs], args.out, settings.seed, args.entities
                )
                if args.bulk_load:
                    measurement["bulk_load"] = await bulk_load_graph(graph, args.out)
                    write_json(args.out / "graph-benchmark.json", measurement)
                return measurement
        if name == "benchmark-vector":
            catalog, key, _ = index_parts(args.dataset, root, providers)
            chunks = list(catalog.values())
            path = root / "vectors" / vector_key(settings, key) / "vectors.npy"
            matrix = await build_vectors(providers, chunks, path)
            questions = select_questions(args.dataset, "dev", args.limit, settings.seed)
            vectors = await providers.embed([question.text for question in questions], query=True)
            return vector_microbenchmark(
                settings,
                chunks,
                matrix,
                vectors,
                backend=args.backend,
                out=args.out,
                ef_values=args.ef,
                warmup=args.warmup,
                repeats=args.repeats,
            )
        if name == "judge-pair":
            left_manifest, left = read_run(args.left)
            right_manifest, right_rows = read_run(args.right)
            if left_manifest["dataset"] != right_manifest["dataset"] or left_manifest[
                "dataset"
            ] != dataset_manifest(args.dataset):
                raise ValueError("pairwise judging needs identical dataset versions")
            if left_manifest["split"] != right_manifest["split"]:
                raise ValueError("pairwise judging needs the same split")
            right = {row["question_id"]: row for row in right_rows}
            if set(right) != {row["question_id"] for row in left}:
                raise ValueError("pairwise judging needs aligned questions")
            question_by_id = {
                q.question_id: q
                for q in select_questions(args.dataset, left_manifest["split"], None, settings.seed)
            }
            comparisons = []
            for row in left:
                q = question_by_id[row["question_id"]]
                other = right[q.question_id]
                if row["status"] == other["status"] == "ok":
                    result = await judge_pair(
                        q.text, row["answer"]["text"], other["answer"]["text"], providers
                    )
                else:
                    result = None
                comparisons.append({"question_id": q.question_id, "judgments": result})
            write_jsonl(args.out, comparisons)
            return {
                "pairs": len(comparisons),
                "scored": sum(item["judgments"] is not None for item in comparisons),
            }
        if name in {"answer", "retrieve", "validate-index"}:
            modes = required_modes(args.mode, getattr(args, "router", "R0"), settings, svm)
            async with open_pipeline(args.dataset, root, providers, modes) as pipeline:
                if name == "answer":
                    return asdict(
                        await pipeline.answer(
                            args.question, mode=args.mode, level=args.router, svm=svm
                        )
                    )
                if name == "retrieve":
                    hits, trace = await pipeline.retrieve(args.question, args.mode)
                    if args.rerank:
                        hits = await pipeline.rerank(args.question, args.mode, hits)
                    return {"hits": [asdict(hit) for hit in hits], "trace": trace}
                if pipeline.graph:
                    await pipeline.graph.snapshot()
                vectors_checked = 0
                if pipeline.vector is not None:
                    keys = list(pipeline.catalog)
                    for start in range(0, len(keys), 256):
                        expected = set(keys[start : start + 256])
                        found = await asyncio.to_thread(pipeline.vector.get_vectors, list(expected))
                        if set(found) != expected:
                            raise ValueError("passage index is missing expected corpus vector IDs")
                        vectors_checked += len(found)
                    if await asyncio.to_thread(pipeline.vector.size) != vectors_checked:
                        raise ValueError("passage index contains unexpected vectors")
                return {
                    "chunks": len(pipeline.catalog),
                    "vectors_checked": vectors_checked,
                    "behavior_hash": pipeline.index_key,
                }
    raise ValueError(f"unsupported operation {name}")


def dispatch(args: argparse.Namespace) -> Any:
    name = args.command
    if name == "environment":
        return {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": dependency_versions(),
        }
    if name == "prepare-fixture":
        return fixture(args.fixtures, args.out)
    if name == "download":
        if args.dataset not in PUBLIC_FILES:
            raise ValueError("this dataset uses upstream access instructions in docs/DATASETS.md")
        expected = json.loads(args.sha256_file.read_text()) if args.sha256_file else {}
        return {
            filename: download(url, args.out / filename, expected.get(filename))
            for filename, url in PUBLIC_FILES[args.dataset].items()
        }
    if name == "prepare":
        return prepare_dataset(args.dataset, args.inputs, args.out, args.seed, args.version)
    if name == "validate-data":
        return dataset_manifest(args.dataset)
    if name == "matrix":
        cases = design(args.block)
        return {"block": args.block, "count": len(cases), "cases": cases}
    if name == "train-router":
        return train_router(args)
    if name == "router-cv":
        manifest = dataset_manifest(args.dataset)
        if args.additional_dev:
            if not args.leave_domain_out:
                raise ValueError("multiple development datasets require leave-one-dataset-out")
            partitions = [
                (path, dataset_manifest(path)) for path in [args.dataset, *args.additional_dev]
            ]
            datasets = [item["dataset"] for _, item in partitions]
            if len(set(datasets)) != len(datasets) or "ragrouter-bench" in datasets:
                raise ValueError(
                    "provide distinct main benchmark datasets for leave-one-dataset-out"
                )
            qs = [
                replace(
                    q,
                    domain=item["dataset"],
                    question_id=f"{item['dataset']}:{q.question_id}",
                    query_type=route_for_query_type(q.query_type),
                )
                for path, item in partitions
                for q in questions_at(path, "dev")
            ]
        else:
            split = "all" if manifest["dataset"] == "ragrouter-bench" else "dev"
            qs = select_questions(args.dataset, split, None, manifest["seed"])
        return cross_validate(qs, args.leave_domain_out, manifest["seed"])
    if name == "calibrate-router":
        initial_manifest, initial = read_run(args.initial)
        fused_manifest, fused = read_run(args.mix)
        if (
            initial_manifest["split"] != "dev"
            or fused_manifest["split"] != "dev"
            or initial_manifest["dataset"] != fused_manifest["dataset"]
            or fused_manifest["mode"] != "mix"
            or initial_manifest["router"] != fused_manifest["router"]
            or initial_manifest["router"] != "R0"
        ):
            raise ValueError("calibration needs paired fixed initial and mix arms from development")
        return calibrate_threshold(initial, fused)
    if name == "audit-agreement":
        return audit_agreement(list(read_jsonl(args.annotations)))
    if name == "export-reference":
        return export_reference(
            args.dataset,
            args.out,
            args.split,
            framework=args.framework,
            limit=args.limit,
            seed=args.seed,
        )
    if name == "run-reference":
        return run_reference(
            args.dataset, args.exported, args.out, args.python, args.settings, timeout=args.timeout
        )
    if name == "import-reference":
        return import_reference(args.dataset, args.exported, args.outcomes, args.version, args.out)
    if name == "export-graphrag-eval":
        run_manifest, outcomes = read_run(args.run)
        return export_graphrag_eval(args.dataset, run_manifest, outcomes, args.out)
    if name == "attach-graphrag-metrics":
        return attach_graphrag_metrics(args.run, args.scores, args.out)
    if name == "report":
        return {"figures": plot_runs(args.runs, args.out)}
    if name == "report-vector-benchmark":
        return {"figure": str(plot_vector_curves(args.csv, args.out))}
    if name == "report-model-study":
        return {"figures": plot_model_study(args.suite, args.out)}
    if name == "compare":
        return compare_runs(args.runs)
    if name == "power":
        left_manifest, left = read_run(args.left)
        right_manifest, right = read_run(args.right)
        if (
            left_manifest["split"] != "dev"
            or right_manifest["split"] != "dev"
            or left_manifest["dataset"] != right_manifest["dataset"]
        ):
            raise ValueError("power analysis requires paired development runs")
        return minimum_detectable_effect(
            {row["question_id"]: row["correct"] for row in left if type(row["correct"]) is bool},
            {row["question_id"]: row["correct"] for row in right if type(row["correct"]) is bool},
            args.sample_size,
        )
    if name == "evaluate":
        manifest = json.loads((args.run / "manifest.json").read_text())
        return report_run(
            args.run,
            list(read_jsonl(args.run / "questions.jsonl")),
            expected=len(manifest["qids"]),
            seed=manifest["configuration"]["seed"],
        )
    if name == "evaluate-router":
        manifest, routed = read_run(args.routed)
        mix_manifest, mix = read_run(args.mix)
        modes = []
        for path in args.static:
            static_manifest, arm = read_run(path)
            if (
                static_manifest["dataset"] != manifest["dataset"]
                or static_manifest["split"] != manifest["split"]
            ):
                raise ValueError("router evaluation arms must share data versions and split")
            modes.extend(arm)
        if (
            mix_manifest["dataset"] != manifest["dataset"]
            or mix_manifest["split"] != manifest["split"]
            or mix_manifest["mode"] != "mix"
        ):
            raise ValueError("savings comparison requires a paired mix run")
        return router_metrics(routed, modes, mix)
    return asyncio.run(operation(args, Settings.read(args.config)))


def main(argv: Sequence[str] | None = None) -> int:
    args = configure().parse_args(argv)
    try:
        result = dispatch(args)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return int(
            args.command == "run" and isinstance(result, dict) and bool(result.get("errors"))
        )
    except (
        OSError,
        KeyError,
        ValueError,
        RuntimeError,
        TypeError,
        ImportError,
        AttributeError,
        httpx.HTTPError,
    ) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
