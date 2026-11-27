"""Result aggregation, paired comparisons and plots generated from run artifacts."""

import csv
import io
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .data import atomic_text, read_jsonl, write_json
from .statistics import holm, interval, mcnemar, paired

QUALITY_FIELDS = (
    "correct",
    "exact_match",
    "token_f1",
    "judge_score",
    "refusal_accuracy",
    "recall_at_5",
    "recall_at_10",
    "recall_at_50",
    "ndcg_at_10",
    "gold_context_recall",
    "gold_context_precision",
    "context_precision",
    "context_recall",
    "faithfulness",
    "evidence_recall",
    "context_relevance",
)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    output = io.StringIO()
    writer = csv.DictWriter(output, fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, output.getvalue())


def report_run(
    out: Path, rows: Sequence[dict[str, Any]], *, expected: int, seed: int
) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {("all", "all"): list(rows)}
    for row in rows:
        for field in ("query_type", "domain"):
            groups.setdefault((field, row[field]), []).append(row)
    summary: list[dict[str, Any]] = []
    for (field, label), group in groups.items():
        descriptor = {
            "group": field,
            "label": label,
            "attempted": len(group),
            "errors": sum(row["status"] == "error" for row in group),
        }
        if field == "all":
            descriptor["expected"] = expected
        for metric in QUALITY_FIELDS:
            measured = [
                float(value) for row in group if (value := row["metrics"].get(metric)) is not None
            ]
            summary.append({**descriptor, "metric": metric, **interval(measured, seed=seed)})
        for metric in (
            "latency",
            "context_tokens",
            "prompt_tokens",
            "completion_tokens",
            "estimated_usd",
        ):
            measured = [float(value) for row in group if (value := row.get(metric)) is not None]
            scale = 1000 if metric == "estimated_usd" else 1
            sample = [value * scale for value in measured]
            entry = {
                **descriptor,
                "metric": "estimated_usd_per_1000" if scale == 1000 else metric,
                **interval(sample, seed=seed),
            }
            if metric == "latency" and sample:
                entry.update(
                    zip(
                        ("p50", "p95", "p99"),
                        map(float, np.quantile(sample, [0.5, 0.95, 0.99])),
                        strict=True,
                    )
                )
            summary.append(entry)
    write_csv(out / "summary.csv", summary)
    complete = len(rows) == expected
    data = {"complete": complete, "expected": expected, "attempted": len(rows), "groups": summary}
    write_json(out / "summary.json", data)
    fields = {
        "correct": "Answer accuracy",
        "recall_at_10": "Retrieval recall@10",
        "token_f1": "Token F1",
        "faithfulness": "Faithfulness",
        "latency": "Latency in seconds",
        "estimated_usd_per_1000": "Estimated USD per 1,000",
    }
    body = [
        "# Evaluation run",
        "",
        f"Attempted {len(rows)} of {expected} questions. {sum(row['status'] == 'error' for row in rows)} failed.",
        "",
        "| Metric | Estimate | 95% interval | Scored |",
        "| --- | ---: | ---: | ---: |",
    ]
    for item in summary:
        if item["group"] != "all" or item["metric"] not in fields:
            continue
        est = "Unavailable" if item["estimate"] is None else f"{item['estimate']:.4f}"
        ci = (
            "Unavailable"
            if item["lower"] is None
            else f"{item['lower']:.4f} to {item['upper']:.4f}"
        )
        body.append(f"| {fields[item['metric']]} | {est} | {ci} | {item['n']} |")
    atomic_text(out / "report.md", "\n".join(body) + "\n")
    return data


def compare_runs(paths: Sequence[Path], *, seed: int = 42) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("a reference and at least one alternative are required")
    reference_manifest = json.loads((paths[0] / "manifest.json").read_text())

    def rows_at(path):
        records = list(read_jsonl(path / "questions.jsonl"))
        indexed = {row["question_id"]: row for row in records}
        if len(indexed) != len(records):
            raise ValueError(f"{path}: duplicate question IDs in comparison outcomes")
        return indexed

    reference_rows = rows_at(paths[0])
    tests: dict[str, Any] = {}
    expected = set(reference_manifest["qids"])
    for directory in paths[1:]:
        manifest = json.loads((directory / "manifest.json").read_text())
        rows = rows_at(directory)
        if (
            manifest["dataset"] != reference_manifest["dataset"]
            or manifest["split"] != reference_manifest["split"]
            or set(manifest["qids"]) != expected
            or len(manifest["qids"]) != len(expected)
        ):
            raise ValueError(f"{directory}: dataset or split versions differ")
        if set(rows) != expected or set(reference_rows) != expected:
            raise ValueError(f"{directory}: paired tests require complete aligned outcomes")
        numeric = {}
        for metric in ("token_f1", "faithfulness", "recall_at_10", "correct"):
            eligible = sorted(
                key
                for key in expected
                if reference_rows[key]["metrics"].get(metric) is not None
                and rows[key]["metrics"].get(metric) is not None
            )
            numeric[metric] = paired(
                {key: float(reference_rows[key]["metrics"][metric]) for key in eligible},
                {key: float(rows[key]["metrics"][metric]) for key in eligible},
                seed=seed,
            )
        eligible = sorted(
            key
            for key in expected
            if type(reference_rows[key]["correct"]) is bool and type(rows[key]["correct"]) is bool
        )
        test = (
            mcnemar(
                {key: reference_rows[key]["correct"] for key in eligible},
                {key: rows[key]["correct"] for key in eligible},
            )
            if eligible
            else None
        )
        tests[manifest["run_id"]] = {
            "effect_sizes": numeric,
            "mcnemar": test,
            "eligible_binary": len(eligible),
        }
    adjusted = holm(
        {key: value["mcnemar"]["p_value"] for key, value in tests.items() if value["mcnemar"]}
    )
    return {
        "reference": reference_manifest["run_id"],
        "comparisons": tests,
        "holm_adjusted_p": adjusted,
    }


def plot_runs(run_paths: Sequence[Path], out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for path in run_paths:
        manifest = json.loads((path / "manifest.json").read_text())
        data = json.loads((path / "summary.json").read_text())
        if not data["complete"]:
            raise ValueError(f"{path}: complete the run before plotting comparisons")
        runs.append((manifest, data["groups"]))

    def select(data: list[dict[str, Any]], metric: str, group: str = "all") -> dict[str, float]:
        return {
            item["label"]: item["estimate"]
            for item in data
            if item["group"] == group and item["metric"] == metric and item["estimate"] is not None
        }

    produced = []
    for statistic, filename, ylabel in (
        ("correct", "accuracy-by-query-type", "Answer accuracy"),
        ("estimated_usd_per_1000", "cost-per-1000", "Estimated USD per 1,000"),
    ):
        groups = [(m["run_id"], select(data, statistic, "query_type")) for m, data in runs]
        labels = sorted({key for _, items in groups for key in items})
        if not labels:
            continue
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.3), 4), layout="constrained")
        width = 0.8 / max(1, len(groups))
        for index, (name, items) in enumerate(groups):
            ax.bar(
                [
                    i - 0.4 + (index + 0.5) * width
                    for i, label in enumerate(labels)
                    if label in items
                ],
                [items[label] for label in labels if label in items],
                width=width,
                label=name,
            )
        ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.legend()
        fig.savefig(out / f"{filename}.png", dpi=160)
        plt.close(fig)
        produced.append(filename)
    points = []
    for manifest, rows in runs:
        accuracy, cost = select(rows, "correct"), select(rows, "estimated_usd_per_1000")
        if "all" in accuracy and "all" in cost:
            points.append((cost["all"], accuracy["all"], manifest["run_id"]))
    if points:
        fig, ax = plt.subplots(layout="constrained")
        for x, y, name in points:
            ax.scatter(x, y, label=name)
        ax.set(xlabel="Estimated USD per 1,000", ylabel="Answer accuracy")
        ax.legend()
        fig.savefig(out / "cost-versus-accuracy.png", dpi=160)
        plt.close(fig)
        produced.append("cost-versus-accuracy")
    categories: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for path in run_paths:
        if (path / "failure-audit.jsonl").exists():
            for row in read_jsonl(path / "failure-audit.jsonl"):
                if cause := row.get("failure_cause"):
                    categories[path.name][cause] += 1
    if categories:
        fig, ax = plt.subplots(layout="constrained")
        labels = sorted({label for causes in categories.values() for label in causes})
        bottom = np.zeros(len(categories))
        for label in labels:
            counts = np.array([causes.get(label, 0) for causes in categories.values()])
            ax.bar(list(categories), counts, bottom=bottom, label=label)
            bottom += counts
        ax.set_ylabel("Annotated failures")
        ax.legend()
        fig.savefig(out / "failure-causes.png", dpi=160)
        plt.close(fig)
        produced.append("failure-causes")
    return produced


def plot_vector_curves(csv_path: Path, out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    with csv_path.open(newline="") as handle:
        records = list(csv.DictReader(handle))
    if not records:
        raise ValueError("benchmark report has no measured vector records")
    groups: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for row in records:
        if row["filtered"] == "False":
            groups[row["backend"], row["concurrency"]].append(
                (float(row["recall_at_10"]), float(row["qps"]))
            )
    if not groups:
        raise ValueError("benchmark report has no unfiltered vector records")
    fig, ax = plt.subplots(layout="constrained")
    for (engine, clients), pairs in sorted(groups.items()):
        recalls, rates = zip(*sorted(pairs), strict=True)
        ax.plot(recalls, rates, marker="o", label=f"{engine}, {clients} clients")
    ax.axvline(0.95, color="gray", linestyle=":")
    ax.set(xlabel="Recall@10 against exact FAISS", ylabel="Queries per second")
    ax.legend()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_model_study(suite: Path, out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    payload = json.loads(suite.read_text())
    if payload["block"] not in {"E5", "E6"} or payload["dry_run"]:
        raise ValueError("model plots require an executed E5 or E6 study")
    out.mkdir(parents=True, exist_ok=True)
    arms = [row for row in payload["arms"] if row.get("median", {}).get("correct") is not None]
    if not arms:
        raise ValueError("model study contains no measured accuracy")
    paths = []
    for dataset in sorted({arm["design"]["dataset"] for arm in arms}):
        selected = [row for row in arms if row["design"]["dataset"] == dataset]
        fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
        if payload["block"] == "E5":
            reference = {
                "embedding": "strong-open",
                "reranking": "open",
                "extraction": "cheap-api",
                "generation": "mid-open",
            }
            baselines = {
                arm["design"]["factor"]: arm["median"]["correct"]
                for arm in selected
                if reference[arm["design"]["factor"]] == arm["design"]["variant"]
            }
            deltas = [
                (
                    f"{arm['design']['factor']} / {arm['design']['variant']}",
                    arm["median"]["correct"] - baselines[arm["design"]["factor"]],
                )
                for arm in selected
                if arm["design"]["factor"] in baselines
                and reference[arm["design"]["factor"]] != arm["design"]["variant"]
            ]
            if not deltas:
                plt.close(fig)
                continue
            labels, values = zip(*deltas, strict=True)
            ax.bar(labels, values)
            ax.tick_params(axis="x", labelrotation=35)
            ax.set_ylabel("Accuracy change from reference")
            stem = "model-factor"
        else:
            tiers = ["small-open", "mid-open", "strong-api"]
            for mode in sorted({arm["design"]["mode"] for arm in selected}):
                data = {
                    arm["design"]["variant"]: arm["median"]["correct"]
                    for arm in selected
                    if arm["design"]["mode"] == mode
                }
                present = [tier for tier in tiers if tier in data]
                ax.plot(present, [data[tier] for tier in present], marker="o", label=mode)
            ax.set_ylabel("Answer accuracy")
            ax.legend()
            stem = "generator-interaction"
        name = f"{stem}-{dataset}.png"
        fig.savefig(out / name, dpi=160)
        plt.close(fig)
        paths.append(name)
    return paths
