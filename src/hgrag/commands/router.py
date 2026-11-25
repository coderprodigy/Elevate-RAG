"""`hgrag router …` commands: RAGRouter-Bench cross-validation and offline E3 evaluation."""

import json
from dataclasses import asdict
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from hgrag.config import PROJECT_ROOT, load_config
from hgrag.data.loaders import REGISTRY
from hgrag.index.models import load_embedder
from hgrag.router.service import evaluate_experiment, ragrouter_cross_validation

router_app = typer.Typer(no_args_is_help=True, help="Router (E3): train, evaluate, compare.")
console = Console()
EVAL_DATASETS = [name for name in REGISTRY if name != "ragrouter_bench"]


@router_app.command("cv")
def cv(
    encoder: Annotated[bool, typer.Option(help="Also the embedding classifier.")] = False,
) -> None:
    """5-fold CV of the R2 query-type classifier on RAGRouter-Bench (free)."""
    config = load_config()
    kinds = ["tfidf", *(["encoder"] if encoder else [])]
    embedder = (
        load_embedder("bge-small", config.retrieval.embedders["bge-small"]) if encoder else None
    )
    for kind in kinds:
        result = ragrouter_cross_validation(config, kind=kind, embedder=embedder)
        console.print(
            f"{kind}: accuracy {result.accuracy:.3f}, macro-F1 {result.macro_f1:.3f} "
            f"({result.folds}-fold, n={result.n})"
        )


@router_app.command("evaluate")
def evaluate(
    experiment: Annotated[str, typer.Option(help="Per-mode experiment (E2) with dev+test runs.")],
    dataset: Annotated[
        list[str] | None, typer.Option("--dataset", "-d", help="Repeatable. Default: all.")
    ] = None,
    encoder: Annotated[bool, typer.Option(help="Add the embedding classifier (R2).")] = False,
    llm: Annotated[bool, typer.Option(help="Add R3 (one LLM call per test question).")] = False,
    yes: Annotated[bool, typer.Option("--yes", help="Confirm R3's API calls.")] = False,
) -> None:
    """Evaluate R0-R4 and the oracle on test, trained/tuned on dev (offline except R3)."""
    config = load_config()
    if llm and not yes:
        console.print(
            "[yellow]R3 makes one router_llm call per test question; re-run with --yes.[/]"
        )
        raise typer.Exit(code=1)
    embedder = (
        load_embedder("bge-small", config.retrieval.embedders["bge-small"]) if encoder else None
    )
    reports = evaluate_experiment(
        config, experiment, dataset or EVAL_DATASETS, embedder=embedder, use_llm=llm
    )
    for report in reports:
        table = Table(title=f"{report.dataset} (test)")
        columns = ("router", "accuracy", "regret", "tokens/q", "$/1k", "saved vs mix", "paths")
        for column in columns:
            table.add_column(column, justify="left" if column in ("router", "paths") else "right")
        for r in report.results:
            table.add_row(
                r.router,
                f"{r.accuracy:.3f}",
                f"{r.regret:.3f}",
                f"{r.tokens_per_query:,.0f}",
                f"{r.usd_per_1k:.3f}",
                f"{r.tokens_saved_vs_baseline:+.1%}",
                ", ".join(f"{p} {s:.0%}" for p, s in r.path_share.items()),
            )
        console.print(table)
    out = PROJECT_ROOT / "reports" / f"router_{experiment}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([asdict(r) for r in reports], indent=2), encoding="utf-8")
    console.print(f"Wrote {out}")
