"""`hgrag` command-line interface. Each sub-app is a thin layer over library code."""

import os
from datetime import date
from typing import Annotated

import typer
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from hgrag.commands.bench import bench_app
from hgrag.commands.data import data_app
from hgrag.commands.eval import eval_app
from hgrag.commands.exp import exp_app
from hgrag.commands.graph import graph_app
from hgrag.commands.index import index_app
from hgrag.commands.report import report_app
from hgrag.commands.router import router_app
from hgrag.config import PROJECT_ROOT, load_config
from hgrag.core.runs import RunContext, new_run_id
from hgrag.llm.gateway import build_gateway
from hgrag.llm.ledger import GroupBy, UsageLedger
from hgrag.logging_setup import configure_logging

app = typer.Typer(no_args_is_help=True, help="Hybrid vector + graph RAG toolkit.")
config_app = typer.Typer(no_args_is_help=True, help="Inspect configuration.")
llm_app = typer.Typer(no_args_is_help=True, help="Model registry and connectivity checks.")
app.add_typer(config_app, name="config")
app.add_typer(llm_app, name="llm")
app.add_typer(data_app, name="data")
app.add_typer(eval_app, name="eval")
app.add_typer(index_app, name="index")
app.add_typer(graph_app, name="graph")
app.add_typer(router_app, name="router")
app.add_typer(bench_app, name="bench")
app.add_typer(exp_app, name="exp")
app.add_typer(report_app, name="report")

console = Console()


@app.callback()
def _main(log_level: Annotated[str, typer.Option(help="Logging level.")] = "INFO") -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.environ.get("HF_TOKEN"):
        # Never fall back to a cached `huggingface-cli login` token: an expired one breaks even
        # public downloads (see docs/setup/03_datasets.md). Public models download anonymously.
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    configure_logging(log_level)


@app.command()
def usage(
    by: Annotated[GroupBy, typer.Option(help="Group spend by this column.")] = GroupBy.ROLE,
    since: Annotated[str | None, typer.Option(help="Only calls on/after YYYY-MM-DD.")] = None,
    run_id: Annotated[str | None, typer.Option(help="Only this run.")] = None,
) -> None:
    """Show API spend recorded in the usage ledger."""
    config = load_config()
    ledger = UsageLedger(config.paths.ledger_db)
    rows = ledger.summarize(by, since=date.fromisoformat(since) if since else None, run_id=run_id)

    table = Table(title=f"API usage by {by.value}")
    for column in (by.value, "calls", "cache hits", "input tok", "output tok", "cost USD"):
        table.add_column(column, justify="left" if column == by.value else "right")
    for row in rows:
        table.add_row(
            row.group,
            f"{row.calls:,}",
            f"{row.cache_hits:,}",
            f"{row.input_tokens:,}",
            f"{row.output_tokens:,}",
            f"{row.cost_usd:.4f}",
        )
    console.print(table)
    total = sum(r.cost_usd for r in rows)
    console.print(
        f"Total: [bold]${total:.4f}[/]  ·  global cap ${config.budget.global_cap_usd:.2f}"
    )


@config_app.command("show")
def config_show() -> None:
    """Print the fully resolved configuration."""
    config = load_config()
    console.print(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))


@llm_app.command("roles")
def llm_roles() -> None:
    """List roles and the model each one uses."""
    models = load_config().models
    table = Table(title="Roles")
    for column in ("role", "model key", "provider", "$/MTok in", "$/MTok out"):
        table.add_column(column)
    for role, key in models.roles.items():
        if key is None:
            table.add_row(role, "[yellow]not set[/]", "", "", "")
            continue
        spec = models.models[key]
        table.add_row(
            role,
            key,
            spec.provider,
            f"{spec.price.input_per_mtok:.2f}",
            f"{spec.price.output_per_mtok:.2f}",
        )
    console.print(table)


@llm_app.command("ping")
def llm_ping(
    role: Annotated[str, typer.Option(help="Role to test.")] = "smoke",
    model: Annotated[
        str | None, typer.Option(help="Model key instead of the role's model.")
    ] = None,
    prompt: Annotated[str, typer.Option(help="Prompt text.")] = "Reply with the single word: pong",
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Force a real call.")] = False,
) -> None:
    """Send one tiny request through the gateway and show tokens and cost."""
    gateway = build_gateway(load_config(), RunContext(run_id=new_run_id("ping"), phase="p0"))
    try:
        response = gateway.complete(
            role, prompt, max_tokens=16, model_key=model, use_cache=not no_cache
        )
    except Exception as error:  # CLI boundary: show a readable message instead of a traceback
        console.print(f"[red]call failed:[/] {type(error).__name__}: {error}")
        console.print("Check the API key in .env (see .env.example) and `hgrag llm roles`.")
        raise typer.Exit(code=1) from None
    console.print(f"[bold]{response.model_key}[/] ({response.provider}) → {response.text!r}")
    console.print(
        f"status={response.status} cache_hit={response.cache_hit} "
        f"in={response.input_tokens} out={response.output_tokens} "
        f"cost=${response.cost_usd:.6f} latency={response.latency_ms:.0f} ms"
    )


if __name__ == "__main__":
    app()
