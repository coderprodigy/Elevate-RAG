"""`elevate-rag` command-line interface (FR-U1)."""

from __future__ import annotations

import asyncio
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import typer

from elevate_rag.utils import configure_logging

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Elevate-RAG: hybrid vector + graph RAG research platform.",
)
data_app = typer.Typer(no_args_is_help=True, help="Download and prepare datasets.")
index_app = typer.Typer(no_args_is_help=True, help="Build, pilot and validate indexes.")
router_app = typer.Typer(no_args_is_help=True, help="Train and evaluate routers.")
exp_app = typer.Typer(no_args_is_help=True, help="Validate, estimate and run experiments.")
bench_app = typer.Typer(no_args_is_help=True, help="Level-A database micro-benchmarks.")
judge_app = typer.Typer(no_args_is_help=True, help="Validate the LLM judges (GroUSE calibration).")
results_app = typer.Typer(no_args_is_help=True, help="Traceable run records in the repository.")
for name, sub in (
    ("data", data_app),
    ("index", index_app),
    ("router", router_app),
    ("experiment", exp_app),
    ("bench", bench_app),
    ("judge", judge_app),
    ("results", results_app),
):
    app.add_typer(sub, name=name)


def _echo(obj: Any) -> None:
    typer.echo(json.dumps(obj, indent=2, default=str))


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    configure_logging("DEBUG" if verbose else "INFO")
    _load_dotenv()


def _load_dotenv() -> None:
    """Read `.env` without overriding variables already set (never writes it)."""
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                if v.strip():
                    os.environ.setdefault(k.strip(), v.strip())


# =============================================================================== env


@app.command("env")
def env_check() -> None:
    """Report Python, key packages, hardware, services and which credentials are present."""
    import importlib.metadata as md

    import psutil

    pk: dict[str, str | None] = {}
    for p in (
        "lightrag-hku",
        "anthropic",
        "numpy",
        "scikit-learn",
        "networkx",
        "tiktoken",
        "sentence-transformers",
        "faiss-cpu",
        "qdrant-client",
        "pymilvus",
        "neo4j",
        "psycopg",
    ):
        try:
            pk[p] = md.version(p)
        except md.PackageNotFoundError:
            pk[p] = None
    docker = shutil.which("docker")
    docker_ok = False
    if docker:
        import subprocess

        # A diagnostic must never crash: a hung or broken daemon is reported as "not available".
        try:
            proc = subprocess.run([docker, "info"], capture_output=True, timeout=20)
            docker_ok = proc.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            docker_ok = False
    try:
        from elevate_rag.utils import Tokenizer

        Tokenizer().count("ok")
        tok_ok = True
    except Exception:
        tok_ok = False
    creds = {
        k: bool(os.getenv(k))
        for k in (
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "OPENAI_COMPAT_BASE_URL",
            "COHERE_API_KEY",
            "JINA_API_KEY",
            "HF_TOKEN",
        )
    }
    _echo(
        {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cpus": os.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
            "disk_free_gb": round(shutil.disk_usage(".").free / 2**30, 1),
            "packages": pk,
            "docker_cli": bool(docker),
            "docker_daemon": docker_ok,
            "tiktoken_encoding_available": tok_ok,
            "credentials_present": creds,
        }
    )


# =============================================================================== data


@data_app.command("list")
def data_list() -> None:
    from elevate_rag.data.datasets import DATASETS

    _echo(
        {
            k: {
                "description": v.description,
                "license": v.license,
                "defaults": v.defaults,
                "files": {
                    f: f"{r.repo_id}/{r.filename}@{r.revision[:10]}" for f, r in v.files.items()
                },
            }
            for k, v in DATASETS.items()
        }
    )


@data_app.command("prepare")
def data_prepare(
    names: list[str] = typer.Argument(..., help="dataset names or 'all'"),
    data_dir: Path = typer.Option(Path("data"), envvar="ELEVATE_DATA_DIR"),
    n_questions: int | None = typer.Option(None, help="override sample size"),
    as_name: str | None = typer.Option(
        None, "--as", help="write the variant under this name (one dataset only)"
    ),
) -> None:
    """Download (pinned revisions), normalise, validate, split and manifest datasets."""
    from elevate_rag.data.datasets import DATASETS
    from elevate_rag.data.prepare import prepare_dataset

    targets = list(DATASETS) if names == ["all"] else names
    if as_name and len(targets) != 1:
        raise typer.BadParameter("--as needs exactly one dataset")
    for n in targets:
        ov = {"n_questions": n_questions} if n_questions else None
        m = prepare_dataset(n, data_dir, ov, out_name=as_name)
        typer.echo(f"{as_name or n}: {m['counts']} version={m['version']}")


@data_app.command("synthetic")
def data_synthetic(data_dir: Path = typer.Option(Path("data"), envvar="ELEVATE_DATA_DIR")) -> None:
    """Write the SYNTHETIC offline test corpus."""
    from elevate_rag.data.synthetic import write_synthetic

    _echo(write_synthetic(data_dir)["counts"])


@data_app.command("info")
def data_info(
    name: str, data_dir: Path = typer.Option(Path("data"), envvar="ELEVATE_DATA_DIR")
) -> None:
    from elevate_rag.data.prepare import DatasetStore

    ds = DatasetStore.open(data_dir, name)
    ds.verify()
    m = ds.manifest
    _echo(
        {
            k: m[k]
            for k in (
                "dataset",
                "version",
                "counts",
                "query_types",
                "answerable",
                "sources",
                "license",
                "synthetic",
            )
        }
    )


# =============================================================================== index


def _cfg(path: Path, overrides: list[str] | None = None) -> Any:
    from elevate_rag.config import load_pipeline_config

    ov = {}
    for o in overrides or []:
        k, v = o.split("=", 1)
        try:
            ov[k] = json.loads(v)
        except json.JSONDecodeError:
            ov[k] = v
    return load_pipeline_config(path, ov)


@index_app.command("build")
def index_build(
    config: Path,
    graph: bool = typer.Option(True, "--graph/--no-graph"),
    set_: list[str] = typer.Option(None, "--set", help="dotted override k=v"),
    limit_docs: int | None = typer.Option(None),
) -> None:
    """Build the shared index (chunks, dense, BM25, LightRAG graph). Resumable."""
    from elevate_rag.data.prepare import DatasetStore
    from elevate_rag.pipeline import ElevateRAG

    cfg = _cfg(config, set_)

    async def go() -> dict[str, Any]:
        ds = DatasetStore.open(cfg.data_dir, cfg.dataset)
        docs = ds.documents()[:limit_docs] if limit_docs else ds.documents()
        sysm = ElevateRAG(cfg)
        try:
            return await sysm.build_index(docs, graph=graph, dataset_version=ds.version)
        finally:
            await sysm.close()

    m = asyncio.run(go())
    _echo(
        {
            k: m[k]
            for k in (
                "index_key",
                "documents",
                "chunks",
                "graph",
                "timings",
                "usage_delta",
                "storage_bytes",
            )
        }
    )


@index_app.command("pilot")
def index_pilot(config: Path, set_: list[str] = typer.Option(None, "--set")) -> None:
    """Index a `budget.pilot_fraction` sample (default 5 %), measure, extrapolate (FR-X5)."""
    import random

    from elevate_rag.data.prepare import DatasetStore
    from elevate_rag.pipeline import ElevateRAG

    cfg = _cfg(config, set_)
    ds = DatasetStore.open(cfg.data_dir, cfg.dataset)
    docs = ds.documents()
    k = max(1, round(len(docs) * cfg.budget.pilot_fraction))
    sample = random.Random(13).sample(docs, k)
    pilot_cfg = cfg.model_copy(update={"index_dir": str(Path(cfg.index_dir) / "_pilot")})

    async def go() -> dict[str, Any]:
        s = ElevateRAG(pilot_cfg)
        try:
            return await s.build_index(sample, graph=True, dataset_version=ds.version)
        finally:
            await s.close()

    m = asyncio.run(go())
    factor = len(docs) / k
    usage = m["usage_delta"]
    _echo(
        {
            "pilot_docs": k,
            "total_docs": len(docs),
            "scale_factor": factor,
            "pilot_usage": usage,
            "pilot_seconds": m["timings"],
            "extrapolated_usd": usage["spent_usd"] * factor,
            "extrapolated_graph_seconds": m["timings"].get("graph_index_s", 0) * factor,
            "note": "extrapolation is linear in document count — an estimate",
        }
    )


@index_app.command("validate")
def index_validate(
    config: Path,
    split: str = typer.Option("dev"),
    set_: list[str] = typer.Option(None, "--set"),
) -> None:
    """Graph quality checks before any QA: answer-entity coverage (gate G3, >= 60 %) and shape."""
    from elevate_rag.data.prepare import DatasetStore
    from elevate_rag.graph.quality import answer_entity_coverage, graph_shape
    from elevate_rag.pipeline import ElevateRAG
    from elevate_rag.utils import write_json

    cfg = _cfg(config, set_)

    async def go() -> dict[str, Any]:
        s = await ElevateRAG(cfg).open()
        try:
            assert s.graph is not None
            assert s.store is not None
            nodes, edges = await s.graph.export_graph()
            ds = DatasetStore.open(cfg.data_dir, cfg.dataset)
            cov = answer_entity_coverage(ds.questions(split), [n["id"] for n in nodes])
            shape = graph_shape(nodes, edges, s.store)
            out = {
                "coverage": cov,
                "shape": shape,
                "gate_G3_coverage_ok": (cov["coverage"] or 0) >= 0.6,
                "fallback_rule": "if coverage < 0.6 after tuning, rebuild with the "
                "LlamaIndex SchemaLLMPathExtractor fallback (FR-G11)",
            }
            write_json(s.index_path / "graph_quality.json", out)
            return out
        finally:
            await s.close()

    _echo(asyncio.run(go()))


@app.command("graph")
def graph_inspect(
    config: Path, entity: str | None = typer.Option(None), limit: int = typer.Option(20)
) -> None:
    """Inspect the graph: top entities by degree, or one entity with its relations."""
    from elevate_rag.pipeline import ElevateRAG

    cfg = _cfg(config)

    async def go() -> Any:
        s = await ElevateRAG(cfg).open()
        try:
            assert s.graph is not None
            nodes, edges = await s.graph.export_graph()
            if entity:
                node = next((n for n in nodes if n["id"] == entity), None)
                rel = [e for e in edges if entity in (e["source"], e["target"])]
                return {"entity": node, "relations": rel[:limit]}
            deg: dict[str, int] = {}
            for e in edges:
                for x in (e["source"], e["target"]):
                    deg[x] = deg.get(x, 0) + 1
            return {
                "nodes": len(nodes),
                "edges": len(edges),
                "top_entities": sorted(deg.items(), key=lambda kv: -kv[1])[:limit],
            }
        finally:
            await s.close()

    _echo(asyncio.run(go()))


# =============================================================================== ask


@app.command("ask")
def ask(
    config: Path,
    question: str,
    mode: str = typer.Option("v4"),
    router: str | None = typer.Option(None, help="static|rules|llm|fallback|classifier"),
    set_: list[str] = typer.Option(None, "--set"),
) -> None:
    """Answer one question with a given mode or router; prints the full query record."""
    from dataclasses import asdict

    from elevate_rag.pipeline import ElevateRAG
    from elevate_rag.routing.fallback import answer_routed
    from elevate_rag.routing.router import Router, TfidfSvmRouter
    from elevate_rag.types import Query

    cfg = _cfg(config, set_)

    async def go() -> Any:
        s = await ElevateRAG(cfg).open()
        try:
            q = Query("cli", question)
            if router:
                from elevate_rag.providers.llm import make_llm

                rc = cfg.router.model_copy(update={"level": router})
                clf = TfidfSvmRouter.load(Path(rc.classifier_path)) if rc.classifier_path else None
                llm = (
                    make_llm(cfg.models.router_llm, role="router", ledger=s.ledger)
                    if cfg.models.router_llm
                    else None
                )
                rec = await answer_routed(s, q, Router(rc, classifier=clf, llm=llm))
            else:
                rec = await s.answer(q, mode)
            return asdict(rec)
        finally:
            await s.close()

    _echo(asyncio.run(go()))


# =============================================================================== router


@router_app.command("train")
def router_train(
    out: Path,
    datasets: list[str] = typer.Option(..., "--dataset"),
    scheme: str = typer.Option("type"),
    split: str = typer.Option("dev", help="never 'test' (FR-D4)"),
    runs: Path | None = typer.Option(None, help="outcome scheme: per-mode runs dir"),
    data_dir: Path = typer.Option(Path("data"), envvar="ELEVATE_DATA_DIR"),
) -> None:
    """Train the R2 TF-IDF + SVM router on dev data (type labels or outcome labels)."""
    from elevate_rag.data.prepare import DatasetStore
    from elevate_rag.routing.router import TfidfSvmRouter

    if split == "test":
        raise typer.BadParameter("training on the test split is not allowed")
    texts, labels = [], []
    if scheme == "type":
        for d in datasets:
            sp = "all" if d == "ragrouter" else split
            for q in DatasetStore.open(data_dir, d).questions(sp):
                texts.append(q.question)
                labels.append(q.query_type)
    else:
        if runs is None:
            raise typer.BadParameter("--runs is required for the outcome scheme")
        texts, labels = _outcome_labels(runs, datasets, data_dir)
    model = TfidfSvmRouter(scheme).fit(texts, labels)
    model.save(out)
    _echo(model.meta)


def _outcome_labels(runs: Path, datasets: list[str], data_dir: Path) -> tuple[list[str], list[str]]:
    from elevate_rag.data.prepare import DatasetStore
    from elevate_rag.reporting.report import load_runs
    from elevate_rag.routing.oracle import outcome_training_labels

    texts, labels = [], []
    for d in datasets:
        per_mode: dict[str, dict[str, dict[str, Any]]] = {}
        for r in load_runs(runs):
            defn = r["manifest"]["definition"]
            if defn["dataset"] != d or not defn["variant"].get("mode"):
                continue
            if defn["split"] == "test":
                raise typer.BadParameter(f"{r['dir']}: outcome labels must come from dev runs")
            rows = [json.loads(x) for x in (r["dir"] / "scored.jsonl").read_text().splitlines()]
            per_mode[defn["variant"]["mode"]] = {
                x["question_id"]: {
                    "correct": bool(x["scores"]["correct"]),
                    "tokens": x["prompt_tokens"] + x["completion_tokens"],
                }
                for x in rows
            }
        qtext = {q.question_id: q.question for q in DatasetStore.open(data_dir, d).questions()}
        for qid, mode in outcome_training_labels(per_mode).items():
            texts.append(qtext[qid])
            labels.append(mode)
    return texts, labels


@router_app.command("cv")
def router_cv(
    datasets: list[str] = typer.Option(..., "--dataset"),
    leave_one_dataset_out: bool = typer.Option(False),
    data_dir: Path = typer.Option(Path("data"), envvar="ELEVATE_DATA_DIR"),
) -> None:
    """5-fold CV (or leave-one-dataset-out) of the type-label R2 router."""
    from elevate_rag.data.prepare import DatasetStore
    from elevate_rag.routing.router import cross_validate

    texts, labels, groups = [], [], []
    for d in datasets:
        sp = "all" if d == "ragrouter" else "dev"
        for q in DatasetStore.open(data_dir, d).questions(sp):
            texts.append(q.question)
            labels.append(q.query_type)
            groups.append(q.metadata.get("domain", d) if d == "ragrouter" else d)
    _echo(cross_validate(texts, labels, groups=groups if leave_one_dataset_out else None))


# =============================================================================== experiments


_PIPELINE_OPT = typer.Option(None, help="run the spec against this pipeline config instead")
_LIMIT_OPT = typer.Option(None, help="at most this many questions per run")
_DATASETS_OPT = typer.Option(None, help="comma-separated subset of the spec's datasets")


def _spec(spec: Path, pipeline: Path | None, limit: int | None, datasets: str | None) -> Any:
    from elevate_rag.experiments.runner import load_spec

    ds = [d.strip() for d in datasets.split(",") if d.strip()] if datasets else None
    return load_spec(spec, pipeline=pipeline, limit=limit, datasets=ds)


@exp_app.command("validate")
def exp_validate(
    spec: Path,
    pipeline: Path | None = _PIPELINE_OPT,
    limit: int | None = _LIMIT_OPT,
    datasets: str | None = _DATASETS_OPT,
) -> None:
    from elevate_rag.experiments.runner import expand, validate

    s, base = _spec(spec, pipeline, limit, datasets)
    runs = expand(s, base)
    problems = validate(s, runs)
    _echo({"runs": len(runs), "problems": problems})
    if problems:
        raise typer.Exit(1)


@exp_app.command("estimate")
def exp_estimate(
    spec: Path,
    pipeline: Path | None = _PIPELINE_OPT,
    limit: int | None = _LIMIT_OPT,
    datasets: str | None = _DATASETS_OPT,
) -> None:
    from elevate_rag.experiments.runner import estimate, expand

    s, base = _spec(spec, pipeline, limit, datasets)
    _echo(estimate(expand(s, base)))


@exp_app.command("run")
def exp_run(
    spec: Path,
    runs_root: Path = typer.Option(Path("runs"), envvar="ELEVATE_RUNS_DIR"),
    dry_run: bool = typer.Option(False),
    only: str | None = typer.Option(None, help="keep runs whose id contains this text"),
    shard: str | None = typer.Option(
        None, help="K/N: run only every N-th run starting at K (0-based), for job arrays"
    ),
    yes: bool = typer.Option(False, help="skip the cost confirmation prompt"),
    pipeline: Path | None = _PIPELINE_OPT,
    limit: int | None = _LIMIT_OPT,
    datasets: str | None = _DATASETS_OPT,
    results_dir: Path = typer.Option(
        Path("results"), envvar="ELEVATE_RESULTS_DIR", help="committed run records go here"
    ),
    record: bool = typer.Option(True, help="write the traceable run record (results/runs/)"),
    mlflow: bool = typer.Option(False, envvar="ELEVATE_MLFLOW", help="also log runs to MLflow"),
    mlflow_uri: str | None = typer.Option(None, help="default: sqlite:///<runs_root>/mlflow.db"),
    ragas: bool | None = typer.Option(
        None, help="score with the ragas package (default: the spec's `ragas` setting)"
    ),
) -> None:
    """Run (or dry-run) an experiment; completed runs are skipped, failed runs resumed.

    Every completed run gets a committed record in <results_dir>/runs/<experiment>/ (git commit,
    config hash, models, metrics with CIs, judge validation) and a line in INDEX.csv."""
    from elevate_rag.experiments.runner import (
        estimate,
        execute_run,
        expand,
        select_shard,
        validate,
    )
    from elevate_rag.tracking import default_tracking_uri

    s, base = _spec(spec, pipeline, limit, datasets)
    if ragas is not None:
        s = s.model_copy(update={"ragas": ragas})
    runs = [r for r in expand(s, base) if not only or only in r.run_id]
    runs = select_shard(runs, shard)
    problems = validate(s, runs)
    est = estimate(runs)
    if dry_run or problems:
        _echo({"plan": [r.run_id for r in runs], "problems": problems, "estimate": est})
        raise typer.Exit(1 if problems else 0)
    if est["total_usd"] > 0 or est["has_unknown_prices"]:
        typer.echo(
            f"Estimated spend ≈ ${est['total_usd']:.2f} "
            f"(unknown prices: {est['has_unknown_prices']})."
        )
        if not yes and not typer.confirm("Proceed?"):
            raise typer.Exit(1)
    uri = (mlflow_uri or default_tracking_uri(runs_root)) if mlflow else None
    failed = []
    for r in runs:
        try:
            summ = asyncio.run(
                execute_run(
                    r,
                    s,
                    runs_root,
                    results_dir=results_dir if record else None,
                    mlflow_uri=uri,
                )
            )
            ov = summ.get("overall", {})
            em, f1, acc = ((ov.get(k) or {}).get("mean") for k in ("em", "f1", "correct"))
            status = (summ.get("judge_validation") or {}).get("status", "no judge")
            typer.echo(f"✓ {r.run_id}: em={em} f1={f1} judge_accuracy={acc} ({status})")
        except Exception as exc:  # recorded in the run manifest; continue with other runs
            failed.append(r.run_id)
            typer.echo(f"✗ {r.run_id}: {type(exc).__name__}: {exc}", err=True)
    if failed:
        typer.echo(f"{len(failed)} run(s) failed: {failed}", err=True)
        raise typer.Exit(1)


@exp_app.command("external")
def exp_external(
    predictions: Path,
    system: str = typer.Option(...),
    dataset: str = typer.Option(...),
    split: str = typer.Option("test"),
    config: Path | None = typer.Option(None, help="pipeline config with models.judge"),
    data_dir: Path = typer.Option(Path("data"), envvar="ELEVATE_DATA_DIR"),
    runs_root: Path = typer.Option(Path("runs"), envvar="ELEVATE_RUNS_DIR"),
) -> None:
    """Score an external system's predictions JSONL with the same evaluator (D-14)."""
    from elevate_rag.experiments.external import ingest_predictions
    from elevate_rag.providers.base import UsageLedger
    from elevate_rag.providers.llm import make_llm

    judge = judge2 = None
    if config is not None:
        cfg = _cfg(config)
        ledger = UsageLedger(max_usd=cfg.budget.max_usd)
        if cfg.models.judge is not None:
            judge = make_llm(cfg.models.judge, role="judge", ledger=ledger)
        if cfg.models.judge2 is not None:
            judge2 = make_llm(cfg.models.judge2, role="judge2", ledger=ledger)
    _echo(
        asyncio.run(
            ingest_predictions(
                predictions,
                system=system,
                dataset=dataset,
                data_dir=data_dir,
                runs_root=runs_root,
                split=split,
                judge_llm=judge,
                judge2_llm=judge2,
            )
        )["overall"]
    )


@exp_app.command("pairwise")
def exp_pairwise(
    run_a: Path,
    run_b: Path,
    config: Path = typer.Option(..., help="pipeline config providing models.judge"),
    data_dir: Path = typer.Option(Path("data"), envvar="ELEVATE_DATA_DIR"),
    out: Path | None = typer.Option(None),
) -> None:
    """Pairwise win rates (comprehensiveness/diversity/empowerment/overall) on global questions,
    judged in both answer orders."""
    from elevate_rag.data.prepare import DatasetStore
    from elevate_rag.evaluation.pairwise import compare_runs
    from elevate_rag.providers.base import UsageLedger
    from elevate_rag.providers.llm import make_llm
    from elevate_rag.utils import write_json

    cfg = _cfg(config)
    if cfg.models.judge is None:
        raise typer.BadParameter("config has no models.judge")
    judge = make_llm(cfg.models.judge, role="judge", ledger=UsageLedger(cfg.budget.max_usd))
    rows = []
    for d in (run_a, run_b):
        rows.append([json.loads(x) for x in (d / "scored.jsonl").read_text().splitlines()])
    ds = json.loads((run_a / "manifest.json").read_text())["definition"]["dataset"]
    res = asyncio.run(
        compare_runs(rows[0], rows[1], DatasetStore.open(data_dir, ds).questions(), judge)
    )
    if out:
        write_json(out, res)
    _echo({k: v for k, v in res.items() if k != "per_question"})


# =============================================================================== bench


@bench_app.command("vector")
def bench_vector(
    config: Path,
    backends: str = typer.Option("exact,faiss"),
    n_queries: int = typer.Option(500),
    out: Path = typer.Option(Path("results/benchmarks/vector.json")),
    container: list[str] = typer.Option(None, help="backend=container for RAM"),
) -> None:
    """Level A: matched-recall vector benchmark on the index's own chunk embeddings."""
    import numpy as np

    from elevate_rag.benchmarks.dbbench import bench_vector_store
    from elevate_rag.pipeline import ElevateRAG
    from elevate_rag.storage.vector import ExactStore
    from elevate_rag.utils import write_json

    cfg = _cfg(config)
    s = ElevateRAG(cfg)
    src = ExactStore.load(cfg.vector_store, s._dense_store_name(), s.index_path / "dense")
    ids, base = src._ids, src._vecs
    rng = np.random.default_rng(13)
    qidx = rng.choice(len(ids), size=min(n_queries, len(ids)), replace=False)
    queries = base[qidx] + rng.normal(0, 0.05, size=(len(qidx), base.shape[1])).astype("float32")
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    cmap = dict(c.split("=", 1) for c in container or [])
    results = []
    for b in backends.split(","):
        vcfg = cfg.vector_store.model_copy(update={"backend": b})
        results.append(
            bench_vector_store(
                _store_factory(vcfg, f"bench_{b}", int(base.shape[1])),
                ids,
                base,
                queries.astype("float32"),
                container=cmap.get(b),
            )
        )
        typer.echo(f"{b}: matched={results[-1]['matched']}")
    write_json(out, results)


def _store_factory(vcfg: Any, name: str, dim: int) -> Any:
    from elevate_rag.storage.vector import make_vector_store

    return lambda: make_vector_store(vcfg, name, dim)


@bench_app.command("graph")
def bench_graph(
    config: Path,
    backends: str = typer.Option("networkx"),
    out: Path = typer.Option(Path("results/benchmarks/graph.json")),
) -> None:
    """Level A: graph-store benchmark on the index's exported LightRAG graph."""
    from elevate_rag.benchmarks.dbbench import bench_graph_store
    from elevate_rag.pipeline import ElevateRAG
    from elevate_rag.storage.graph import make_graph_store
    from elevate_rag.utils import write_json

    cfg = _cfg(config)

    async def export() -> Any:
        s = await ElevateRAG(cfg).open()
        try:
            assert s.graph is not None
            return await s.graph.export_graph()
        finally:
            await s.close()

    nodes, edges = asyncio.run(export())
    gn = [
        {
            "id": n["id"],
            "type": n.get("entity_type", ""),
            "description": str(n.get("description", ""))[:500],
        }
        for n in nodes
    ]
    ge = [
        {
            "src": e["source"],
            "tgt": e["target"],
            "description": str(e.get("description", ""))[:500],
            "weight": float(e.get("weight") or 1.0),
        }
        for e in edges
    ]
    results = []
    for b in backends.split(","):
        store = make_graph_store(cfg.graph_store.model_copy(update={"backend": b}))
        try:
            results.append(bench_graph_store(store, gn, ge))
        finally:
            store.close()
    write_json(out, results)
    _echo(results)


# =============================================================================== judge


@judge_app.command("calibrate")
def judge_calibrate(
    config: Path,
    set_: list[str] = typer.Option(None, "--set"),
    results_dir: Path = typer.Option(Path("results"), envvar="ELEVATE_RESULTS_DIR"),
    data_dir: Path = typer.Option(Path("data"), envvar="ELEVATE_DATA_DIR"),
    grouse_file: Path | None = typer.Option(None, help="local GroUSE test.jsonl (offline)"),
    concurrency: int = typer.Option(4),
) -> None:
    """Score each configured judge on GroUSE (published, expert-written expected verdicts).

    Writes results/judge_calibration/<model@revision@judge-version>.json, which every later run
    reads to decide whether its judge-based accuracies count as validated."""
    from elevate_rag.evaluation.judge_calibration import (
        calibrate,
        download_grouse,
        load_cases,
        write_calibration,
    )
    from elevate_rag.providers.base import UsageLedger
    from elevate_rag.providers.llm import make_llm

    cfg = _cfg(config, set_)
    roles = [(r, m) for r, m in (("judge", cfg.models.judge), ("judge2", cfg.models.judge2)) if m]
    if not roles:
        raise typer.BadParameter("the pipeline config defines no models.judge / models.judge2")
    path = grouse_file or download_grouse(data_dir / "raw")
    cases = load_cases(path)
    out_dir = results_dir / "judge_calibration"

    async def go() -> list[dict[str, Any]]:
        ledger = UsageLedger(max_usd=cfg.budget.max_usd)
        res = []
        for role, m in roles:
            llm = make_llm(m, role=role, ledger=ledger)
            r = await calibrate(llm, cases, concurrency=concurrency)
            written = write_calibration(out_dir, r)
            res.append(
                {
                    "role": role,
                    "model": r["model"],
                    "n": r["n"],
                    "pass_rate": r["pass_rate"],
                    "calibrated": r["calibrated"],
                    "written": str(written),
                }
            )
        return res

    _echo(asyncio.run(go()))


# =============================================================================== results


@results_app.command("sync")
def results_sync(
    runs_root: Path = typer.Option(Path("runs"), envvar="ELEVATE_RUNS_DIR"),
    results_dir: Path = typer.Option(Path("results"), envvar="ELEVATE_RESULTS_DIR"),
    experiment: str | None = typer.Option(None),
    mlflow: bool = typer.Option(False, envvar="ELEVATE_MLFLOW", help="also log to MLflow"),
    mlflow_uri: str | None = typer.Option(None, help="default: sqlite:///<runs_root>/mlflow.db"),
) -> None:
    """(Re)write the committed record of every completed run, e.g. after copying runs back
    from a cluster or EC2, re-checking judge validation against the calibration files present
    now. Records and INDEX.csv go to <results_dir>/runs/."""
    from elevate_rag.evaluation.judge_calibration import revalidate
    from elevate_rag.reporting.report import load_runs
    from elevate_rag.tracking import default_tracking_uri, rebuild_index, record_run
    from elevate_rag.utils import write_json

    uri = (mlflow_uri or default_tracking_uri(runs_root)) if mlflow else None
    written = 0
    for r in load_runs(runs_root, experiment):
        if r["manifest"].get("demo"):
            continue
        summary = r["summary"]
        config = r["manifest"]["definition"].get("config", {})
        jv = revalidate(summary, config, results_dir / "judge_calibration")
        if jv is not None and jv != summary.get("judge_validation"):
            summary["judge_validation"] = jv
            write_json(r["dir"] / "summary.json", summary)
        record_run(r["manifest"], summary, r["dir"], results_dir, mlflow_uri=uri)
        written += 1
    index = rebuild_index(results_dir)
    typer.echo(f"{written} run record(s) written; index: {index}")


# =============================================================================== report


@app.command("report")
def report(
    runs_root: Path = typer.Option(Path("runs"), envvar="ELEVATE_RUNS_DIR"),
    out: Path = typer.Option(Path("results/report")),
    reference: str = typer.Option("V4"),
    experiment: str | None = typer.Option(None),
    vector_bench: Path | None = typer.Option(None),
) -> None:
    """Aggregate completed runs into CSV tables, paired tests, figures and a Markdown report."""
    from elevate_rag.reporting import report as R

    runs = R.load_runs(runs_root, experiment)
    if not runs:
        raise typer.BadParameter(f"no completed runs under {runs_root}")
    table = R.results_table(runs)
    R.write_csv(out / "results.csv", table, R.TABLE_FIELDS)
    comps = R.paired_comparisons(runs, reference)
    if comps:
        R.write_csv(out / "paired_comparisons.csv", comps, list(comps[0]))
    figs = []
    for ds in sorted({r["manifest"]["definition"]["dataset"] for r in runs}):
        for fn, stem in (
            (R.fig_accuracy_by_type, "accuracy_by_type"),
            (R.fig_router_scatter, "cost_vs_accuracy"),
        ):
            p = fn(runs, ds, out / f"{stem}_{ds}.png")
            if p:
                figs.append(p)
    em_comps = [c for c in comps if c["metric"] == "em"]
    if em_comps and (
        p := R.fig_factor_importance(
            em_comps, out / "factor_importance.png", any(t["demo"] for t in table)
        )
    ):
        figs.append(p)
    bench_rows = (
        json.loads(vector_bench.read_text()) if vector_bench and vector_bench.exists() else []
    )
    if p := R.fig_recall_qps(bench_rows, out / "recall_qps.png"):
        figs.append(p)
    R.write_markdown_report(out / "REPORT.md", table, comps, figs)
    typer.echo(f"report written to {out}")


# =============================================================================== demo


@app.command("demo")
def demo(out: Path = typer.Option(Path("demo_output"))) -> None:
    """Offline end-to-end demo on the SYNTHETIC corpus (no credentials, no network)."""
    from elevate_rag.demo import run_demo

    _echo(asyncio.run(run_demo(out)))


if __name__ == "__main__":
    app()
