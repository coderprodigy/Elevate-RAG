"""`hgrag exp …` commands: list, plan (dry run with costs and blockers) and run experiments."""

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from hgrag.config import load_config
from hgrag.data.loaders import REGISTRY
from hgrag.eval.experiments import list_experiments, load_experiment
from hgrag.eval.planner import PlannedRun, plan_experiment, run_planned

exp_app = typer.Typer(no_args_is_help=True, help="Experiments: plan and run whole E1-E6 blocks.")
console = Console()
EVAL_DATASETS = [name for name in REGISTRY if name != "ragrouter_bench"]

ExpOption = Annotated[str, typer.Option("--exp", help="Experiment name (configs/experiments/).")]
SplitOption = Annotated[str | None, typer.Option(help="Override the experiment's split.")]


@exp_app.command("list")
def list_() -> None:
    """Experiments and their arms."""
    for name in list_experiments():
        experiment = load_experiment(name)
        arms = experiment.expand()
        frozen = " [frozen]" if experiment.frozen else ""
        console.print(
            f"[bold]{name}[/]{frozen}: {len(arms)} arm(s); {experiment.description.strip()}"
        )


def _plan(name: str, split: str | None) -> list[PlannedRun]:
    config = load_config()
    experiment = load_experiment(name)
    problems = experiment.violations()
    if problems:
        console.print(f"[red]protocol violations:[/] {problems}")
        raise typer.Exit(code=1)
    planned = plan_experiment(
        config, experiment, datasets=experiment.datasets or EVAL_DATASETS, split=split
    )
    table = Table(title=f"{name} ({split or experiment.split})")
    for column in ("arm", "pipeline", "dataset", "pending", "max USD", "status"):
        table.add_column(column, justify="right" if column in ("pending", "max USD") else "left")
    for run in planned:
        table.add_row(
            run.label,
            run.arm.pipeline,
            run.dataset,
            str(run.prepared.plan.n_pending) if run.prepared else "-",
            f"{run.cost:.4f}" if run.prepared else "-",
            "ready" if run.prepared else f"[yellow]blocked[/]: {run.blocked}",
        )
    console.print(table)
    ready = [r for r in planned if r.prepared]
    console.print(
        f"{len(ready)}/{len(planned)} runs ready · upper-bound cost of ready runs: "
        f"[bold]${sum(r.cost for r in ready):.4f}[/]"
    )
    return planned


@exp_app.command("plan")
def plan(exp: ExpOption, split: SplitOption = None) -> None:
    """Dry run: every arm x dataset with pending questions, cost bound and blockers."""
    _plan(exp, split)


@exp_app.command("run")
def run(
    exp: ExpOption,
    split: SplitOption = None,
    yes: Annotated[bool, typer.Option("--yes", help="Confirm above the threshold.")] = False,
) -> None:
    """Run every ready arm x dataset (resumable; blocked runs are skipped and listed)."""
    planned = _plan(exp, split)
    config = load_config()
    total = sum(r.cost for r in planned)
    threshold = config.budget.confirm_above_usd
    if total > threshold and not yes:
        console.print(f"[yellow]Above the ${threshold:.2f} threshold: re-run with --yes.[/]")
        raise typer.Exit(code=1)
    for result, outcome, metrics in run_planned(config, planned):
        correct = metrics.overall.metrics["correct"]
        console.print(
            f"{result.label} / {result.dataset}: +{outcome.n_new}, failed {outcome.n_failed}, "
            f"correct {correct.mean:.3f} [{correct.low:.3f}, {correct.high:.3f}]"
            + (f" [red]stopped: {outcome.stopped_reason}[/]" if outcome.stopped_reason else "")
        )
