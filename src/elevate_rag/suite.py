"""Build one-factor experiment arms from a reference configuration."""

import json
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import median
from typing import Any

from .config import Model, Settings
from .data import stable_hash
from .experiments import design, plan_run, run
from .routing import SVM


@dataclass(frozen=True)
class Arm:
    case: dict[str, Any]
    dataset: Path
    settings: Settings
    mode: str
    router: str = "R0"
    classifier: SVM | None = None
    repeats: int = 1
    warmup: int = 0
    limit: int | None = None

    @property
    def identifier(self) -> str:
        return f"{self.case['block'].lower()}-{self.case['dataset']}-{stable_hash(self.case, self.settings.fingerprint)[:12]}"


def prepare_arms(
    spec: Path,
    block: str,
    reference: Settings,
) -> tuple[list[Arm], list[dict[str, Any]]]:
    with spec.open("rb") as stream:
        options = tomllib.load(stream)
    arms, excluded = [], []
    for case in design(block):
        dataset = case["dataset"]
        if dataset not in options.get("datasets", {}):
            continue
        factor, variant = case["factor"], case["variant"]
        mode, router, model = case["mode"], "R0", None
        chosen = reference
        repeats, warmup, limit = 1, 0, None
        if (
            factor == "external-framework"
            or case.get("level") == "A"
            or case.get("folds")
            or variant == "offline-oracle"
        ):
            excluded.append({**case, "reason": "run this case through its separate workflow"})
            continue
        if factor == "router":
            router = variant.partition("-")[0]
            if router == "R2":
                path = options.get("routers", {}).get(dataset, {}).get(variant)
                if not path:
                    excluded.append(
                        {**case, "reason": "requires an aligned development classifier"}
                    )
                    continue
                model = SVM.read(spec.parent / path)
        elif factor == "storage":
            if variant in {"qdrant", "milvus", "pgvector", "faiss"}:
                chosen = replace(reference, graph_vector_store=variant, vector_store=variant)
            else:
                chosen = replace(reference, graph_store=variant)
            repeats, warmup, limit = case["repetitions"], case["warmup"], case["questions"]
        elif factor in {
            "embedding",
            "reranking",
            "extraction",
            "generation",
            "generation-interaction",
        }:
            role = "generation" if factor == "generation-interaction" else factor
            table = options.get("models", {}).get(role, {}).get(variant)
            if table is None:
                excluded.append(
                    {**case, "reason": "model tier requires a frozen provider definition"}
                )
                continue
            model_fields = dict(table)
            tokenizer = model_fields.pop("context_tokenizer_model", None)
            tokenizer_revision = model_fields.pop("context_tokenizer_revision", None)
            alternative = Model(**model_fields)
            if (
                role == "generation"
                and not reference.synthetic
                and (
                    alternative.identity != reference.model("generation").identity
                    or tokenizer is not None
                    or tokenizer_revision is not None
                )
            ):
                if not tokenizer or not tokenizer_revision:
                    raise ValueError(f"generation tier {variant} requires a frozen chat tokenizer")
                chosen = replace(
                    reference,
                    context_tokenizer_model=tokenizer,
                    context_tokenizer_revision=tokenizer_revision,
                    models={**reference.models, role: alternative},
                )
            else:
                if tokenizer is not None or tokenizer_revision is not None:
                    raise ValueError("chat tokenizer overrides require a real generation tier")
                chosen = replace(reference, models={**reference.models, role: alternative})
        elif block == "ablations":
            chosen = replace(reference, **{factor: variant})
        if block == "E1" and mode == "V4":
            chosen.model("reranking")
        arms.append(
            Arm(
                case,
                spec.parent / options["datasets"][dataset],
                chosen,
                mode,
                router,
                model,
                repeats,
                warmup,
                limit,
            )
        )
    if not arms and not excluded:
        raise ValueError("suite defines no datasets used by this block")
    return arms, excluded


async def execute_suite(
    arms: list[Arm],
    *,
    split: str,
    root: Path,
    results: Path,
    dry_run: bool,
    retry_failed: bool,
) -> list[dict[str, Any]]:
    output = []
    for arm in arms:
        uncached = arm.case.get("level") == "B"
        payload, _ = plan_run(
            arm.dataset,
            arm.settings,
            mode=arm.mode,
            router=arm.router,
            split=split,
            limit=arm.limit,
            uncached=uncached,
            svm=arm.classifier,
            warmup=arm.warmup,
        )
        if payload["dataset"]["dataset"] != arm.case["dataset"]:
            raise ValueError(f"dataset manifest does not match suite arm {arm.identifier}")
        row: dict[str, Any] = {
            "design": arm.case,
            "id": arm.identifier,
            "configuration_hash": payload["configuration_hash"],
            "questions": len(payload["qids"]),
            "repetitions": arm.repeats,
            "generation_only_max_usd": payload["generation_only_max_usd"] * arm.repeats,
        }
        if not dry_run:
            result_paths, summaries = [], []
            for repetition in range(1, arm.repeats + 1):
                path = await run(
                    arm.dataset,
                    root,
                    results,
                    arm.settings,
                    mode=arm.mode,
                    router=arm.router,
                    split=split,
                    limit=arm.limit,
                    run_id=f"{arm.identifier}-{split}-{repetition}",
                    uncached=uncached,
                    svm=arm.classifier,
                    retry_failed=retry_failed,
                    warmup=arm.warmup,
                )
                result_paths.append(str(path))
                summary = json.loads((path / "summary.json").read_text())
                fields = {
                    item["metric"]: item["estimate"]
                    for item in summary["groups"]
                    if item["group"] == "all"
                }
                latency = next(
                    item
                    for item in summary["groups"]
                    if item["group"] == "all" and item["metric"] == "latency"
                )
                fields.update(
                    {f"latency_{field}": latency.get(field) for field in ("p50", "p95", "p99")}
                )
                summaries.append(fields)
            row["paths"] = result_paths
            row["median"] = {
                key: median(item[key] for item in summaries)
                if all(item[key] is not None for item in summaries)
                else None
                for key in summaries[0]
            }
        output.append(row)
    return output
