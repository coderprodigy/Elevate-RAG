"""Configuration-driven experiment runner (FR-X1, FR-X2, FR-X4, FR-X5).

An experiment spec (YAML) names a base pipeline config, datasets and *variants*. Each variant
changes one factor (a mode, a router level, or dotted config overrides) relative to the
reference configuration. The runner expands spec × datasets into runs with deterministic IDs,
validates them, estimates cost, and executes them resumably:

  runs/<experiment>/<run_id>/manifest.json      immutable run definition (+ status)
                            predictions.jsonl  per-question online records
                            scored.jsonl       + offline metrics
                            summary.json       aggregates with 95 % CIs
                            ledger.json        every model call (cost/tokens, cached flag)
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from elevate_rag import __version__
from elevate_rag.config import (
    PipelineConfig,
    ProviderConfig,
    apply_overrides,
    load_yaml_with_extends,
)
from elevate_rag.data.prepare import DatasetStore
from elevate_rag.graph.lightrag_engine import GRAPH_MODES
from elevate_rag.providers.base import FREE_PROVIDERS, UsageLedger
from elevate_rag.utils import ConfigError, ElevateError, stable_hash, write_json

log = logging.getLogger(__name__)

EXTRACTOR_DATASETS = ("multihop_rag", "graphrag_bench_medical")  # plan: re-indexing is costly


class Variant(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    mode: str | None = None
    router: dict[str, Any] | None = None
    overrides: dict[str, Any] = Field(default_factory=dict)
    tune_on_dev: bool = False  # R4: tune the fallback threshold on the dev split first


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    question: str
    description: str = ""
    pipeline: str
    datasets: list[str]
    variants: list[Variant]
    split: str = "test"
    limit: int | None = None
    judge: bool = True
    llm_metrics: bool = False
    concurrency: int = 4
    retry_failed: bool = False


@dataclass
class Run:
    run_id: str
    experiment: str
    dataset: str
    variant: Variant
    cfg: PipelineConfig
    split: str
    limit: int | None

    @property
    def needs_graph(self) -> bool:
        if self.variant.mode in GRAPH_MODES:
            return True
        r = self.cfg.router
        if self.variant.router is None:
            return False
        modes = {
            r.static_mode,
            r.fallback_cheap_mode,
            r.fallback_escalate_mode,
            *r.type_to_mode.values(),
            *r.modes,
        }
        return bool(modes & set(GRAPH_MODES))


def load_spec(
    path: Path | str,
    *,
    pipeline: Path | str | None = None,
    limit: int | None = None,
    datasets: list[str] | None = None,
) -> tuple[ExperimentSpec, Path]:
    """Load an experiment spec, optionally re-targeted without editing it: another pipeline
    config (e.g. the cluster one), a per-run question limit, or a subset of its datasets.
    Run ids are derived from the resulting config, so re-targeted runs never collide."""
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    if pipeline is not None:
        data["pipeline"] = str(Path(pipeline).resolve())
    if limit is not None:
        data["limit"] = limit
    if datasets:
        unknown = sorted(set(datasets) - set(data.get("datasets", [])))
        if unknown:
            raise ConfigError(f"{path.name}: datasets {unknown} are not in this experiment")
        data["datasets"] = [d for d in data["datasets"] if d in set(datasets)]
    return ExperimentSpec.model_validate(data), path.parent


def expand(spec: ExperimentSpec, base_dir: Path) -> list[Run]:
    base = load_yaml_with_extends(base_dir / spec.pipeline)
    runs = []
    for ds in spec.datasets:
        for v in spec.variants:
            data = apply_overrides(base, {"dataset": ds, **v.overrides})
            if v.router is not None:
                data = apply_overrides(data, {f"router.{k}": val for k, val in v.router.items()})
            try:
                cfg = PipelineConfig.model_validate(data)
            except ValueError as exc:
                raise ConfigError(f"{spec.id}/{v.name}/{ds}: {exc}") from exc
            ident = {
                "cfg": cfg.config_hash(),
                "mode": v.mode,
                "router": v.router,
                "split": spec.split,
                "limit": spec.limit,
                "tune": v.tune_on_dev,
            }
            run_id = f"{spec.id}-{v.name}-{ds}-{stable_hash(ident, 8)}"
            runs.append(Run(run_id, spec.id, ds, v, cfg, spec.split, spec.limit))
    return runs


def select_shard(runs: list[Run], shard: str | None) -> list[Run]:
    """``"K/N"`` → every N-th run starting at K (0-based). The order of `expand` is deterministic,
    so N parallel jobs with K = 0..N-1 cover every run exactly once."""
    if not shard:
        return runs
    try:
        k, n = (int(x) for x in shard.split("/"))
    except ValueError as exc:
        raise ConfigError(f"--shard must look like K/N, got {shard!r}") from exc
    if n < 1 or not 0 <= k < n:
        raise ConfigError(f"--shard {shard!r}: need 0 <= K < N")
    return runs[k::n]


# ------------------------------------------------------------------------------- validation


def _family(p: ProviderConfig | None) -> str | None:
    return p.family if p is not None else None


def validate(spec: ExperimentSpec, runs: list[Run], *, check_data: bool = True) -> list[str]:
    """Returns a list of problems; empty = valid. Never auto-fixes."""
    problems: list[str] = []
    if spec.split not in ("test", "dev", "all"):
        problems.append(f"split must be test/dev/all, got {spec.split!r}")
    gen_families = {_family(r.cfg.models.generator) for r in runs}
    for r in runs:
        tag = f"{r.run_id}"
        if (r.variant.mode is None) == (r.variant.router is None):
            problems.append(f"{tag}: a variant needs exactly one of `mode` or `router`")
        m = r.cfg.models
        if spec.judge:
            if m.judge is None:
                problems.append(f"{tag}: models.judge must be set (no default judge — D-15)")
            elif m.judge.family is None or m.judge.family in gen_families:
                problems.append(
                    f"{tag}: judge family {m.judge.family!r} must be set and differ "
                    f"from every generator family {sorted(map(str, gen_families))}"
                )
        if "models.extractor" in r.variant.overrides and r.dataset not in EXTRACTOR_DATASETS:
            problems.append(
                f"{tag}: extractor comparisons are restricted to {EXTRACTOR_DATASETS} (plan, FR-X2)"
            )
        if r.needs_graph and m.extractor is None:
            problems.append(f"{tag}: graph modes need models.extractor")
        if (
            r.variant.router
            and r.variant.router.get("level") == "classifier"
            and not r.cfg.router.classifier_path
        ):
            problems.append(
                f"{tag}: classifier router needs router.classifier_path "
                "(train it on dev data with `elevate-rag router train`)"
            )
        paid = [
            p
            for p in (
                m.embedder,
                m.reranker,
                m.extractor,
                m.keyword,
                m.generator,
                m.router_llm,
                m.judge,
            )
            if p
            and p.provider not in FREE_PROVIDERS
            and (p.price_input_per_mtok is None or p.price_input_per_mtok > 0)
        ]
        if paid and r.cfg.budget.max_usd == 0:
            problems.append(
                f"{tag}: uses paid/unknown-price providers "
                f"{sorted({p.model for p in paid})} but budget.max_usd is 0"
            )
        if check_data:
            try:
                DatasetStore.open(r.cfg.data_dir, r.dataset).verify()
            except ElevateError as exc:
                problems.append(f"{tag}: {exc}")
    return problems


# ------------------------------------------------------------------------------- estimation


def estimate(runs: list[Run]) -> dict[str, Any]:
    """Pessimistic *estimate* (flagged as such) of tokens and USD before running (FR-X5).

    Query cost: (context budget + 400 prompt overhead) input, 300 output per generation; graph
    modes add one keyword call (≈ 600 in / 60 out); judge ≈ 400 in / 60 out per question.
    Indexing cost (graph runs, counted once per index): chunk tokens × (1 + gleaning) passes ×
    (chunk + ≈ 2,000 prompt) input and ≈ 500 output per pass — validate with the 5 % pilot."""
    seen_index: set[str] = set()
    rows, total = [], 0.0
    unknown = False
    for r in runs:
        try:
            ds = DatasetStore.open(r.cfg.data_dir, r.dataset)
            n_q = len(ds.questions(r.split))
            counts = ds.manifest["counts"]
        except ElevateError:
            n_q, counts = 0, {"documents": 0}
        if r.limit:
            n_q = min(n_q, r.limit)
        m, rc = r.cfg.models, r.cfg.retrieval
        gen_in = (rc.context_budget + 400) * n_q
        gen_out = 300 * n_q
        usd = _usd(m.generator, gen_in, gen_out)
        if r.needs_graph and (m.keyword or m.extractor):
            usd = _add(usd, _usd(m.keyword or m.extractor, 600 * n_q, 60 * n_q))
        if m.judge:
            usd = _add(usd, _usd(m.judge, 400 * n_q, 60 * n_q))
        idx_usd: float | None = 0.0
        key = f"{r.dataset}:{r.cfg.index_key()}"
        if r.needs_graph and key not in seen_index and m.extractor:
            seen_index.add(key)
            # rough corpus size: documents × mean chunk tokens is unknown before chunking;
            # use the manifest's document count × chunk size as an upper bound.
            n_chunks = counts.get("documents", 0) * 1.2
            passes = 1 + r.cfg.lightrag.entity_extract_max_gleaning
            idx_usd = _usd(
                m.extractor,
                int(n_chunks * passes * (r.cfg.chunking.chunk_tokens + 2000)),
                int(n_chunks * passes * 500),
            )
        run_usd = _add(usd, idx_usd)
        unknown |= run_usd is None
        total += run_usd or 0.0
        rows.append(
            {
                "run_id": r.run_id,
                "questions": n_q,
                "query_usd": usd,
                "index_usd": idx_usd,
                "is_estimate": True,
            }
        )
    return {"runs": rows, "total_usd": total, "has_unknown_prices": unknown, "is_estimate": True}


def _usd(p: ProviderConfig | None, tin: int, tout: int) -> float | None:
    if p is None or p.provider in FREE_PROVIDERS:
        return 0.0
    if p.price_input_per_mtok is None:
        return None
    return (tin * p.price_input_per_mtok + tout * (p.price_output_per_mtok or 0)) / 1e6


def _add(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a + b


# ------------------------------------------------------------------------------- execution


def run_dir(runs_root: Path, run: Run) -> Path:
    return runs_root / run.experiment / run.run_id


def write_manifest(
    d: Path, run: Run, spec: ExperimentSpec, status: str, extra: dict[str, Any] | None = None
) -> None:
    mf = d / "manifest.json"
    definition = {
        "run_id": run.run_id,
        "experiment": spec.id,
        "research_question": spec.question,
        "dataset": run.dataset,
        "variant": run.variant.model_dump(),
        "split": run.split,
        "limit": run.limit,
        "config_hash": run.cfg.config_hash(),
        "index_key": run.cfg.index_key(),
        "config": run.cfg.model_dump(mode="json"),
    }
    if mf.exists():
        old = json.loads(mf.read_text())
        if old["definition"]["config_hash"] != definition["config_hash"]:
            raise ElevateError(
                f"{mf}: run definition changed — manifests are immutable; use a new variant name"
            )
        created = old["created_at"]
    else:
        created = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    write_json(
        mf,
        {
            "definition": definition,
            "created_at": created,
            "status": status,
            "updated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "environment": {
                "elevate_rag": __version__,
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            "demo": run.cfg.demo,
            **(extra or {}),
        },
    )


async def execute_run(run: Run, spec: ExperimentSpec, runs_root: Path) -> dict[str, Any]:
    from elevate_rag.evaluation.harness import (
        run_questions,
        score_rows,
        summarize,
        write_scored,
    )
    from elevate_rag.pipeline import ElevateRAG
    from elevate_rag.providers.llm import make_llm
    from elevate_rag.routing.fallback import answer_routed

    d = run_dir(runs_root, run)
    mf = d / "manifest.json"
    if (
        mf.exists()
        and json.loads(mf.read_text()).get("status") == "complete"
        and not spec.retry_failed
    ):
        log.info("skip %s (complete)", run.run_id)
        return json.loads((d / "summary.json").read_text())
    d.mkdir(parents=True, exist_ok=True)
    write_manifest(d, run, spec, "running")
    ds = DatasetStore.open(run.cfg.data_dir, run.dataset)
    ds.verify()
    questions = ds.questions(run.split)
    if run.limit:
        questions = questions[: run.limit]
    ledger = UsageLedger(max_usd=run.cfg.budget.max_usd)
    system = ElevateRAG(run.cfg, ledger=ledger)
    try:
        try:
            await system.open(graph=run.needs_graph)
        except ElevateError:
            log.info("building index for %s", run.run_id)
            await system.build_index(
                ds.documents(), graph=run.needs_graph, dataset_version=ds.version
            )
            await system.close()
            system = ElevateRAG(run.cfg, ledger=ledger)
            await system.open(graph=run.needs_graph)
        router = None
        if run.variant.router is not None:
            router = await _make_router(run, system, ds, ledger, runs_root)
        meta = {
            "run_id": run.run_id,
            "dataset": run.dataset,
            "dataset_version": ds.version,
            "config_hash": run.cfg.config_hash(),
            "variant": run.variant.name,
            "generator_model": run.cfg.models.generator.model,
            "embedder_model": system.embedder.model_id,
        }

        async def answer(q: Any) -> Any:
            if router is not None:
                return await answer_routed(system, q, router)
            return await system.answer(q, run.variant.mode or "v4")

        rows = await run_questions(
            answer,
            questions,
            d / "predictions.jsonl",
            run_meta=meta,
            concurrency=spec.concurrency,
            retry_failed=spec.retry_failed,
        )
        judge_llm = (
            make_llm(
                run.cfg.models.judge, role="judge", ledger=ledger, cache=system.generator_llm.cache
            )
            if spec.judge and run.cfg.models.judge
            else None
        )
        scored = await score_rows(
            rows,
            questions,
            system.store,
            judge_llm=judge_llm,
            metrics_llm=judge_llm if spec.llm_metrics else None,
            concurrency=spec.concurrency,
            string_evidence_valid=ds.string_evidence_valid,
        )
        write_scored(d / "scored.jsonl", scored)
        summary = summarize(scored)
        summary["ledger"] = ledger.summary()
        write_json(d / "summary.json", summary)
        write_json(d / "ledger.json", ledger.as_dicts())
        write_manifest(d, run, spec, "complete", {"n_questions": len(questions)})
        return summary
    except Exception as exc:
        write_manifest(d, run, spec, "failed", {"error": f"{type(exc).__name__}: {exc}"[:2000]})
        raise
    finally:
        await system.close()


async def _make_router(
    run: Run, system: Any, ds: DatasetStore, ledger: UsageLedger, runs_root: Path
) -> Any:
    from elevate_rag.evaluation.harness import score_rows
    from elevate_rag.providers.llm import make_llm
    from elevate_rag.routing.fallback import tune_threshold
    from elevate_rag.routing.router import Router, TfidfSvmRouter

    rcfg = run.cfg.router
    clf = TfidfSvmRouter.load(Path(rcfg.classifier_path)) if rcfg.classifier_path else None
    llm = (
        make_llm(
            run.cfg.models.router_llm,
            role="router",
            ledger=ledger,
            cache=system.generator_llm.cache,
        )
        if run.cfg.models.router_llm
        else None
    )
    if rcfg.level == "fallback" and run.variant.tune_on_dev:
        dev = ds.questions("dev")
        cheap = [await system.answer(q.to_query(), rcfg.fallback_cheap_mode) for q in dev]
        esc = [await system.answer(q.to_query(), rcfg.fallback_escalate_mode) for q in dev]
        from dataclasses import asdict

        sc = await score_rows([asdict(r) for r in cheap], dev, system.store)
        se = await score_rows([asdict(r) for r in esc], dev, system.store)
        tuned = tune_threshold(
            [r.top_score for r in cheap],
            [bool(s["scores"]["correct"]) for s in sc],
            [bool(s["scores"]["correct"]) for s in se],
            [r.refused for r in cheap],
            on_refusal=rcfg.fallback_on_refusal,
        )
        write_json(
            run_dir(runs_root, run) / "fallback_tuning.json",
            {
                **tuned,
                "dev_questions": len(dev),
                "note": "tuned with EM/refusal correctness on the dev split only",
            },
        )
        rcfg = rcfg.model_copy(update={"fallback_threshold": tuned["threshold"]})
    return Router(rcfg, classifier=clf, llm=llm)
