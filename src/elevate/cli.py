"""Command-line entry point for Elevate-RAG.

Every phase, experiment and exit check is reached through the ``elevate`` command.
Subcommands are stubs in Phase 0 scaffolding and are filled in by later tasks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from elevate import __version__

app = typer.Typer(
    name="elevate",
    help="Elevate-RAG research harness: vector, graph and routed retrieval.",
    no_args_is_help=True,
    add_completion=False,
)

# Distinct from 1 (check failure) and 2 (Click usage error).
NOT_IMPLEMENTED_EXIT_CODE = 3


def _not_implemented(command: str) -> None:
    """Report that a subcommand is not wired yet and exit with a non-zero status."""
    typer.echo(f"'{command}' is not implemented yet.", err=True)
    raise typer.Exit(code=NOT_IMPLEMENTED_EXIT_CODE)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"elevate {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """Elevate-RAG research harness."""


@app.command("verify-setup")
def verify_setup() -> None:
    """Check Python, dependencies, env vars, datasets, local models and Docker."""
    _not_implemented("verify-setup")


@app.command("exit-check")
def exit_check(
    phase: Annotated[int, typer.Option("--phase", min=0, max=7, help="Phase number (0-7).")],
) -> None:
    """Run the exit check for one phase on the Tiny_Sample."""
    _not_implemented(f"exit-check --phase {phase}")


@app.command("run")
def run(
    pipeline: Annotated[str, typer.Argument(help="Pipeline name, e.g. V4, local, mix, R4.")],
    config: Annotated[Path | None, typer.Option("--config", help="Experiment config file.")] = None,
) -> None:
    """Run one configured pipeline over one dataset split."""
    _not_implemented(f"run {pipeline}")


@app.command("experiment")
def experiment(
    file: Annotated[Path, typer.Argument(help="Declarative experiment file (E1-E6, A1-A4).")],
) -> None:
    """Execute an experiment matrix from a declarative config file."""
    _not_implemented(f"experiment {file}")


@app.command("e2e")
def e2e(
    tiny: Annotated[
        bool, typer.Option("--tiny", help="Run on the bundled Tiny_Sample with no-setup backends.")
    ] = False,
    config: Annotated[
        Path | None, typer.Option("--config", help="Experiment config for a full-dataset run.")
    ] = None,
) -> None:
    """Run the full pipeline from raw data to report assets."""
    _not_implemented("e2e")


if __name__ == "__main__":
    app()
