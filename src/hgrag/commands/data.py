"""`hgrag data …` commands: download raw files, build processed datasets, write data cards."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from hgrag.config import PROJECT_ROOT, load_config
from hgrag.data.build import build_dataset
from hgrag.data.datacard import DataCard, render_markdown
from hgrag.data.download import download_dataset
from hgrag.data.io import dataset_files, raw_dir
from hgrag.data.loaders import REGISTRY, resolve

data_app = typer.Typer(no_args_is_help=True, help="Datasets: download, build, data cards.")
console = Console()

DatasetOption = Annotated[
    list[str],
    typer.Option("--dataset", "-d", help=f"Dataset name, repeatable, or 'all'. {list(REGISTRY)}"),
]
REPORT_PATH = PROJECT_ROOT / "reports" / "data_cards.md"


@data_app.command("download")
def download(dataset: DatasetOption = ["all"]) -> None:  # noqa: B006 (Typer needs a literal default)
    """Download raw files (free; about 400 MB in total). Existing files are skipped."""
    config = load_config()
    for spec in resolve(dataset):
        paths = download_dataset(spec, raw_dir(config))
        console.print(f"[green]✓[/] {spec.name}: {len(paths)} file(s)")


@data_app.command("build")
def build(dataset: DatasetOption = ["all"]) -> None:  # noqa: B006
    """Build data/processed/<name>/ (corpus, queries, frozen splits, data card)."""
    config = load_config()
    for spec in resolve(dataset):
        card = build_dataset(spec, config)
        console.print(
            f"[green]✓[/] {spec.name}: {card.n_documents:,} docs, {card.n_queries:,} queries "
            f"({card.n_dev:,} dev / {card.n_test:,} test)"
        )


@data_app.command("card")
def card(
    output: Annotated[Path, typer.Option(help="Markdown file to write.")] = REPORT_PATH,
) -> None:
    """Summarise every built dataset and write the markdown data cards for the report."""
    config = load_config()
    cards = [
        DataCard.model_validate_json(files.datacard.read_text(encoding="utf-8"))
        for name in REGISTRY
        if (files := dataset_files(config, name)).datacard.is_file()
    ]
    if not cards:
        console.print("[yellow]No built datasets yet. Run: hgrag data build[/]")
        raise typer.Exit(code=1)

    table = Table(title="Data cards")
    for column in ("dataset", "docs", "~tokens", "queries", "dev", "test", "unanswerable"):
        table.add_column(column, justify="left" if column == "dataset" else "right")
    for c in cards:
        table.add_row(
            c.name,
            f"{c.n_documents:,}",
            f"{c.approx_corpus_tokens:,}",
            f"{c.n_queries:,}",
            f"{c.n_dev:,}",
            f"{c.n_test:,}",
            f"{c.n_unanswerable:,}",
        )
    console.print(table)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(cards), encoding="utf-8")
    console.print(f"Wrote {output}")
