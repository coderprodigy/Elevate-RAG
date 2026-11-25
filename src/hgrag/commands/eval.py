"""`hgrag eval …` commands: run pipelines over dataset splits and score them."""

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from hgrag.config import ConfigError, load_config
from hgrag.data.loaders import REGISTRY
from hgrag.eval.batch import read_models
from hgrag.eval.human_labels import KAPPA_THRESHOLD, HumanLabel, cohen_kappa, labels_path
from hgrag.eval.runner import SplitLockedError
from hgrag.eval.scoring import RunMetrics
from hgrag.eval.service import (
    CONTEXT_METRIC_FILES,
    PipelineOptions,
    execute,
    execute_context_metric,
    execute_judging,
    experiment_run_dirs,
    judge_verdicts,
    label_sample,
    prepare_judging,
    prepare_runs,
    repeat_groups,
    score_run_dir,
)
from hgrag.pipelines.registry import PIPELINES

eval_app = typer.Typer(no_args_is_help=True, help="Evaluation: run pipelines and score them.")
console = Console()

EVAL_DATASETS = [name for name in REGISTRY if name != "ragrouter_bench"]  # router-only data


@eval_app.command("run")
def run(
    pipeline: Annotated[str, typer.Option(help=f"One of {sorted(PIPELINES)}.")],
    experiment: Annotated[str, typer.Option(help="Experiment name (configs/experiments/).")],
    dataset: Annotated[
        list[str] | None, typer.Option("--dataset", "-d", help="Repeatable. Default: all.")
    ] = None,
    split: Annotated[
        str, typer.Option(help="dev or test (test needs a frozen experiment).")
    ] = "dev",
    limit: Annotated[int | None, typer.Option(help="Only the first N questions.")] = None,
    model: Annotated[str | None, typer.Option(help="Override the generator model key.")] = None,
    chunker: Annotated[str | None, typer.Option(help="Override the chunker (ablation).")] = None,
    embedder: Annotated[str | None, typer.Option(help="Override the embedder (E5).")] = None,
    reranker: Annotated[str | None, typer.Option(help="Override the reranker (E5).")] = None,
    repeat: Annotated[
        int | None, typer.Option(help="Variance check: uncached repeat N (own run folder).")
    ] = None,
    extractor: Annotated[
        str | None, typer.Option(help="Graph pipelines: extraction model key (E5).")
    ] = None,
    graph_backend: Annotated[
        str | None, typer.Option(help="Graph pipelines: configs/graph.yaml backend (E4 B).")
    ] = None,
    concurrency: Annotated[int, typer.Option(help="Parallel requests.")] = 8,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show the cost plan only.")] = False,
    yes: Annotated[bool, typer.Option("--yes", help="Confirm runs above the threshold.")] = False,
) -> None:
    """Answer questions with a pipeline (resumable), then score the run."""
    if split not in ("dev", "test"):
        raise typer.BadParameter("split must be 'dev' or 'test'")
    config = load_config()
    datasets = dataset or EVAL_DATASETS
    try:
        runs = prepare_runs(
            config,
            experiment=experiment,
            datasets=datasets,
            split=split,  # type: ignore[arg-type]
            pipeline_name=pipeline,
            limit=limit,
            model_key=model,
            concurrency=concurrency,
            options=PipelineOptions(
                chunker=chunker,
                embedder=embedder,
                reranker=reranker,
                repeat=repeat,
                extractor=extractor,
                graph_backend=graph_backend,
            ),
        )
    except (SplitLockedError, ConfigError, KeyError, FileNotFoundError) as error:
        console.print(f"[red]cannot start:[/] {error}")
        raise typer.Exit(code=1) from None

    table = Table(title=f"Run plan: {pipeline} on {split}")
    for column in ("dataset", "questions", "done", "pending", "max cost USD"):
        table.add_column(column, justify="left" if column == "dataset" else "right")
    for r in runs:
        p = r.plan
        table.add_row(
            r.spec.dataset,
            str(p.n_selected),
            str(p.n_done),
            str(p.n_pending),
            f"{p.max_cost_usd:.4f}",
        )
    total = sum(r.plan.max_cost_usd for r in runs)
    console.print(table)
    console.print(f"Upper-bound cost: [bold]${total:.4f}[/] (cached answers count as $0)")

    if dry_run:
        return
    threshold = config.budget.confirm_above_usd
    if total > threshold and not yes:
        console.print(
            f"[yellow]Above the ${threshold:.2f} confirm threshold: re-run with --yes.[/]"
        )
        raise typer.Exit(code=1)

    for r in runs:
        outcome, metrics = execute(config, r)
        status = f"[red]stopped: {outcome.stopped_reason}[/]" if outcome.stopped_reason else "ok"
        console.print(
            f"{r.spec.dataset}: +{outcome.n_new} answers, {outcome.n_failed} failed, {status}"
        )
        _print_metrics(r.spec.run_name, metrics)


@eval_app.command("score")
def score(experiment: Annotated[str, typer.Option(help="Experiment to (re)score.")]) -> None:
    """Recompute scores.jsonl and metrics.json for every run of an experiment."""
    config = load_config()
    run_dirs = experiment_run_dirs(config, experiment)
    if not run_dirs:
        console.print(f"[yellow]No runs under runs/{experiment}/[/]")
        raise typer.Exit(code=1)
    for out_dir in run_dirs:
        _print_metrics(out_dir.name, score_run_dir(config, out_dir))


def _print_metrics(title: str, metrics: RunMetrics) -> None:
    table = Table(title=f"{title}  (primary: {metrics.primary_metric})")
    for column in ("group", "n", "correct [95% CI]", "EM", "F1", "refusal acc", "$ / 1k q"):
        table.add_column(column, justify="left" if column == "group" else "right")
    groups = {"overall": metrics.overall, **metrics.by_query_type}
    for name, group in groups.items():
        m = group.metrics

        def mean(key: str, m: dict = m) -> str:  # type: ignore[type-arg]
            return f"{m[key].mean:.3f}" if key in m else "-"

        correct = m["correct"]
        table.add_row(
            name,
            str(group.n),
            f"{correct.mean:.3f} [{correct.low:.3f}, {correct.high:.3f}]",
            mean("exact_match"),
            mean("f1"),
            mean("refusal_accuracy"),
            f"{group.usd_per_1k_queries:.3f}",
        )
    if metrics.n_missing_predictions:
        console.print(
            f"[yellow]{metrics.n_missing_predictions} questions have no prediction yet[/]"
        )
    console.print(table)


@eval_app.command("judge")
def judge(
    experiment: Annotated[str, typer.Option(help="Judge every run of this experiment.")],
    model: Annotated[str | None, typer.Option(help="Override the judge model key.")] = None,
    concurrency: Annotated[int, typer.Option(help="Parallel requests.")] = 8,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show the cost plan only.")] = False,
    yes: Annotated[bool, typer.Option("--yes", help="Confirm runs above the threshold.")] = False,
) -> None:
    """LLM-judge correctness for every answered question (resumable), then re-score."""
    config = load_config()
    try:
        jobs = prepare_judging(config, experiment_run_dirs(config, experiment), model_key=model)
    except (ConfigError, KeyError) as error:
        console.print(f"[red]cannot start:[/] {error}")
        raise typer.Exit(code=1) from None
    total = sum(job.max_cost_usd for job in jobs)
    for job in jobs:
        console.print(f"{job.out_dir.name}: {len(job.pending)} to judge, ≤ ${job.max_cost_usd:.4f}")
    console.print(f"Upper-bound judge cost: [bold]${total:.4f}[/]")
    if dry_run or not _confirmed(total, yes, config.budget.confirm_above_usd):
        return
    for job in jobs:
        outcome = execute_judging(config, job, concurrency=concurrency)
        console.print(f"{job.out_dir.name}: +{outcome.n_new} judgments, {outcome.n_failed} failed")
        _print_metrics(job.out_dir.name, score_run_dir(config, job.out_dir))


@eval_app.command("context-metrics")
def context_metrics(
    experiment: Annotated[str, typer.Option(help="Score every retrieval run of this experiment.")],
    metric: Annotated[
        str, typer.Option(help="faithfulness | evidence_recall (GraphRAG-Bench retrieval).")
    ] = "faithfulness",
    limit: Annotated[int, typer.Option(help="Answers per run (plan: 200).")] = 200,
    model: Annotated[str | None, typer.Option(help="Override the judge model key.")] = None,
    yes: Annotated[
        bool, typer.Option("--yes", help="Confirm (1-2 judge calls per answer).")
    ] = False,
) -> None:
    """LLM-checked context metrics: faithfulness of answers, or recall of gold evidence."""
    if metric not in CONTEXT_METRIC_FILES:
        raise typer.BadParameter(f"metric must be one of {sorted(CONTEXT_METRIC_FILES)}")
    config = load_config()
    if not yes:
        console.print(f"[yellow]Makes up to {2 * limit} judge calls per run. Re-run with --yes.[/]")
        raise typer.Exit(code=1)
    for out_dir in experiment_run_dirs(config, experiment):
        outcome = execute_context_metric(
            config,
            out_dir,
            metric,  # type: ignore[arg-type]  # validated against CONTEXT_METRIC_FILES above
            limit=limit,
            model_key=model,
        )
        console.print(f"{out_dir.name}: +{outcome.n_new} scored, {outcome.n_failed} failed")


@eval_app.command("label")
def label(
    experiment: Annotated[str, typer.Option(help="Sample answers from this experiment.")],
    labeller: Annotated[str, typer.Option(help="Your name (one label file per person).")],
    n: Annotated[int, typer.Option(help="Sample size (plan: 100).")] = 100,
) -> None:
    """Blind human labelling for judge validation. Resumable; type q to stop."""
    config = load_config()
    path = labels_path(config.paths.runs_dir / experiment, labeller)
    done = {item.item_id for item in read_models(path, HumanLabel)}
    items = [item for item in label_sample(config, experiment, n) if item.item_id not in done]
    console.print(
        f"{len(done)} labelled, {len(items)} to go. Keys: c=correct p=partial w=wrong q=quit"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    choices = {"c": "correct", "p": "partial", "w": "wrong"}
    for index, item in enumerate(items, len(done) + 1):
        console.rule(f"{index}/{n}")
        console.print(f"[bold]Question:[/] {item.question}")
        console.print(f"[bold]Reference:[/] {' | '.join(item.references)}")
        console.print(f"[bold]Answer:[/] {item.answer}")
        key = typer.prompt("verdict [c/p/w/q]").strip().lower()
        while key not in (*choices, "q"):
            key = typer.prompt("please type c, p, w or q").strip().lower()
        if key == "q":
            break
        row = HumanLabel(
            item_id=item.item_id,
            run=item.run,
            qid=item.qid,
            labeller=labeller,
            verdict=choices[key],
        )
        with path.open("a", encoding="utf-8") as handle:
            handle.write(row.model_dump_json() + "\n")


@eval_app.command("kappa")
def kappa(experiment: Annotated[str, typer.Option(help="Experiment with human labels.")]) -> None:
    """Cohen's kappa between labellers and between each labeller and the LLM judge."""
    config = load_config()
    label_dir = config.paths.runs_dir / experiment / "labels"
    labels = {
        f.stem: {(lab.run, lab.qid): lab.verdict for lab in read_models(f, HumanLabel)}
        for f in sorted(label_dir.glob("*.jsonl"))
    }
    raters = {**labels, "judge": judge_verdicts(config, experiment)}
    names = sorted(raters)
    table = Table(title=f"Cohen's kappa (pass ≥ {KAPPA_THRESHOLD})")
    for column in ("rater A", "rater B", "shared items", "kappa", "pass"):
        table.add_column(column)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = sorted(raters[a].keys() & raters[b].keys())
            if len(shared) < 2:
                continue
            value = cohen_kappa([raters[a][k] for k in shared], [raters[b][k] for k in shared])
            table.add_row(
                a, b, str(len(shared)), f"{value:.3f}", "✓" if value >= KAPPA_THRESHOLD else "✗"
            )
    console.print(table)


def _confirmed(total: float, yes: bool, threshold: float) -> bool:
    if total > threshold and not yes:
        console.print(
            f"[yellow]Above the ${threshold:.2f} confirm threshold: re-run with --yes.[/]"
        )
        raise typer.Exit(code=1)
    return True


@eval_app.command("variance")
def variance(
    experiment: Annotated[str, typer.Option(help="Experiment with --repeat runs.")],
) -> None:
    """Run-to-run variance across uncached repeats (needed because Claude has no temperature 0)."""
    config = load_config()
    groups = repeat_groups(config, experiment)
    if not groups:
        console.print("[yellow]No repeat runs found (use eval run --repeat 1/2/3).[/]")
        raise typer.Exit(code=1)
    table = Table(title="Run-to-run variance")
    for column in ("run", "repeats", "questions", "accuracy per repeat", "sd", "identical answers"):
        table.add_column(column)
    for name, report in groups.items():
        table.add_row(
            name,
            str(report.n_repeats),
            str(report.n_questions),
            ", ".join(f"{a:.3f}" for a in report.accuracies),
            f"{report.accuracy_sd:.4f}",
            f"{100 * report.identical_answer_rate:.1f}%",
        )
    console.print(table)
