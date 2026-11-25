"""`hgrag report …` commands: build report assets, compare systems, error analysis."""

from typing import Annotated

import typer
from rich.console import Console

from hgrag.analysis.collect import load_experiment_runs
from hgrag.analysis.compare import compare_to_reference
from hgrag.analysis.errors import CAUSES, FailureItem, FailureTag, sample_failures
from hgrag.analysis.report import build_report
from hgrag.analysis.tables import comparison_table
from hgrag.config import PROJECT_ROOT, load_config
from hgrag.data.io import dataset_files, load_queries
from hgrag.eval.batch import read_models
from hgrag.eval.runner import load_predictions

report_app = typer.Typer(no_args_is_help=True, help="Report: tables, figures, comparisons, errors.")
console = Console()
REPORTS = PROJECT_ROOT / "reports"
ExpOption = Annotated[str, typer.Option("--exp", help="Experiment name.")]


@report_app.command("build")
def build() -> None:
    """Regenerate all tables (reports/tables) and figures (reports/figures) from runs/."""
    written = build_report(load_config(), REPORTS)
    console.print(f"Wrote {len(written)} files under {REPORTS}")


@report_app.command("compare")
def compare(
    exp: ExpOption,
    reference: Annotated[str, typer.Option(help="Reference system.")] = "v4",
    split: Annotated[str, typer.Option(help="dev or test.")] = "test",
) -> None:
    """Paired tests (McNemar + bootstrap, Holm-corrected) of every system vs the reference."""
    rows = compare_to_reference(load_experiment_runs(load_config(), exp, split), reference)
    console.print(comparison_table(rows) if rows else "[yellow]nothing to compare[/]")


@report_app.command("errors-sample")
def errors_sample(
    exp: ExpOption,
    n: Annotated[int, typer.Option(help="Failures per system (plan: 50).")] = 50,
    split: Annotated[str, typer.Option(help="dev or test.")] = "test",
) -> None:
    """Sample failures per system with a suggested cause, for tagging."""
    config = load_config()
    items = [
        item
        for run in load_experiment_runs(config, exp, split)
        for item in sample_failures(
            run, n, config.seed, config.retrieval.search.context_token_budget
        )
    ]
    path = config.paths.runs_dir / exp / "errors" / "sample.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(i.model_dump_json() + "\n" for i in items), encoding="utf-8")
    console.print(f"Wrote {len(items)} failures to {path}")


@report_app.command("errors-tag")
def errors_tag(exp: ExpOption, labeller: Annotated[str, typer.Option(help="Your name.")]) -> None:
    """Confirm or change the suggested cause of each failure. Enter = accept, q = quit."""
    config = load_config()
    folder = config.paths.runs_dir / exp / "errors"
    items = read_models(folder / "sample.jsonl", FailureItem)
    path = folder / f"{labeller}.jsonl"
    done = {(t.run, t.qid) for t in read_models(path, FailureTag)}
    menu = "  ".join(f"{i}={c}" for i, c in enumerate(CAUSES, 1))
    for item in items:
        if (item.run, item.qid) in done:
            continue
        dataset = item.run.split("__")[0]
        query = next(q for q in load_queries(dataset_files(config, dataset)) if q.qid == item.qid)
        prediction = next(
            p
            for p in load_predictions(config.paths.runs_dir / exp / item.run / "predictions.jsonl")
            if p.qid == item.qid
        )
        console.rule(f"{item.run} · {item.query_type}")
        gold = " | ".join(query.answers) or "(unanswerable)"
        console.print(f"[bold]Q:[/] {query.question}\n[bold]Gold:[/] {gold}")
        console.print(f"[bold]Answer:[/] {prediction.answer}")
        console.print(f"[dim]{len(prediction.contexts)} context units[/]")
        console.print(f"Suggested: [cyan]{item.suggested}[/]   {menu}")
        key = typer.prompt("cause [Enter/1-6/q]", default="").strip().lower()
        if key == "q":
            break
        cause = (
            CAUSES[int(key) - 1]
            if key.isdigit() and 1 <= int(key) <= len(CAUSES)
            else item.suggested
        )
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                FailureTag(
                    run=item.run, qid=item.qid, labeller=labeller, cause=cause
                ).model_dump_json()
                + "\n"
            )
