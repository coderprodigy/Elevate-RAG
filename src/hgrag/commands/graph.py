"""`hgrag graph …` commands: cost-first graph builds, quality checks, triple audit."""

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from hgrag.config import AppConfig, load_config
from hgrag.data.io import dataset_files, load_queries
from hgrag.data.loaders import REGISTRY
from hgrag.eval.batch import read_models
from hgrag.eval.human_labels import KAPPA_THRESHOLD, cohen_kappa
from hgrag.graph.audit import TripleItem, TripleLabel, audit_dir, sample_triples
from hgrag.graph.build import build_graph, default_extractor, estimate_build
from hgrag.graph.lightrag_adapter import GraphLocation
from hgrag.graph.quality import COVERAGE_GATE, answer_entity_coverage, graph_shape, load_graph
from hgrag.index.build import default_chunker
from hgrag.index.models import load_embedder

graph_app = typer.Typer(no_args_is_help=True, help="Graph RAG: build, quality checks, audit.")
console = Console()
CORPUS_DATASETS = [name for name in REGISTRY if name != "ragrouter_bench"]

DatasetOption = Annotated[
    list[str] | None, typer.Option("--dataset", "-d", help="Repeatable. Default: all.")
]
ExtractorOption = Annotated[
    str | None, typer.Option(help="Extraction model key (default: `extractor` role).")
]
EmbedderOption = Annotated[str | None, typer.Option(help="Default: reference embedder.")]
BackendOption = Annotated[str | None, typer.Option(help="configs/graph.yaml backend.")]


def _location(
    config: AppConfig,
    dataset: str,
    extractor: str | None,
    embedder: str | None,
    backend: str | None,
    sample: bool,
) -> GraphLocation:
    return GraphLocation(
        dataset=dataset,
        chunker=default_chunker(config, dataset),
        extractor=extractor or default_extractor(config),
        embedder=embedder or config.retrieval.reference_embedder,
        backend=backend or config.graph.reference_backend,
        sample=sample,
    )


@graph_app.command("estimate")
def estimate(dataset: DatasetOption = None, extractor: ExtractorOption = None) -> None:
    """Upper-bound indexing cost per dataset (no calls). Run `graph build --sample` next."""
    config = load_config()
    key = extractor or default_extractor(config)
    table = Table(title=f"Graph indexing cost upper bound ({key})")
    for column in ("dataset", "chunks", "full ≤ USD", "5% sample chunks", "sample ≤ USD"):
        table.add_column(column, justify="left" if column == "dataset" else "right")
    total = 0.0
    for name in dataset or CORPUS_DATASETS:
        e = estimate_build(config, name, default_chunker(config, name), key)
        total += e.upper_bound_usd
        table.add_row(
            name,
            f"{e.n_chunks:,}",
            f"{e.upper_bound_usd:.2f}",
            f"{e.sample_chunks:,}",
            f"{e.sample_upper_bound_usd:.3f}",
        )
    console.print(table)
    console.print(
        f"All datasets, full build: ≤ [bold]${total:.2f}[/] (upper bound; measure with --sample)"
    )


@graph_app.command("build")
def build(
    dataset: DatasetOption = None,
    sample: Annotated[
        bool, typer.Option("--sample", help="Index 5% of chunks; measure cost.")
    ] = False,
    extractor: ExtractorOption = None,
    embedder: EmbedderOption = None,
    backend: BackendOption = None,
    yes: Annotated[
        bool, typer.Option("--yes", help="Confirm a build above the threshold.")
    ] = False,
) -> None:
    """Index a dataset with LightRAG over our chunks (resumable). Sample first, then full."""
    config = load_config()
    datasets = dataset or CORPUS_DATASETS
    key = extractor or default_extractor(config)
    bound = sum(
        (lambda e: e.sample_upper_bound_usd if sample else e.upper_bound_usd)(
            estimate_build(config, name, default_chunker(config, name), key)
        )
        for name in datasets
    )
    console.print(f"Upper-bound extraction cost: ${bound:.3f} ({'sample' if sample else 'full'})")
    threshold = config.budget.confirm_above_usd
    if bound > threshold and not yes:
        console.print(f"[yellow]Above the ${threshold:.2f} threshold: re-run with --yes.[/]")
        raise typer.Exit(code=1)
    embedder_name = embedder or config.retrieval.reference_embedder
    model = load_embedder(embedder_name, config.retrieval.embedders[embedder_name])
    for name in datasets:
        location = _location(config, name, key, embedder_name, backend, sample)
        manifest = build_graph(config, location, embedder=model)
        extra = (
            f", extrapolated full build ≈ ${manifest.extrapolated_full_usd:.2f}"
            if manifest.extrapolated_full_usd is not None
            else ""
        )
        console.print(
            f"[green]✓[/] {name}: {manifest.n_chunks:,} chunks, status {manifest.status_counts}, "
            f"spent ${manifest.spent_usd:.4f} in {manifest.seconds:.0f}s{extra}"
        )


@graph_app.command("quality")
def quality(
    dataset: DatasetOption = None,
    extractor: ExtractorOption = None,
    embedder: EmbedderOption = None,
    sample: Annotated[bool, typer.Option("--sample", help="Check the sample build.")] = False,
) -> None:
    """Answer-entity coverage (gate G3: ≥ 60%) and graph shape, per built dataset."""
    config = load_config()
    table = Table(title=f"Graph quality (coverage gate ≥ {COVERAGE_GATE:.0%})")
    for column in (
        "dataset",
        "nodes",
        "edges",
        "avg deg",
        "clustering",
        "dup rate",
        "coverage",
        "gate",
    ):
        table.add_column(column, justify="left" if column == "dataset" else "right")
    for name in dataset or CORPUS_DATASETS:
        location = _location(config, name, extractor, embedder, "local", sample)
        try:
            graph = load_graph(location.working_dir(config))
        except FileNotFoundError:
            table.add_row(name, *["-"] * 6, "not built")
            continue
        shape = graph_shape(graph)
        coverage = answer_entity_coverage(graph, load_queries(dataset_files(config, name)))
        table.add_row(
            name,
            f"{shape.nodes:,}",
            f"{shape.edges:,}",
            f"{shape.average_degree:.2f}",
            f"{shape.clustering:.3f}",
            f"{shape.duplicate_rate:.3f}",
            f"{coverage.rate:.1%} ({coverage.covered}/{coverage.eligible})",
            "✓" if coverage.passes_gate else "✗",
        )
    console.print(table)


@graph_app.command("audit")
def audit(
    dataset: Annotated[str, typer.Option(help="Dataset whose graph to audit.")],
    n: Annotated[int, typer.Option(help="Triples to sample (plan: 50).")] = 50,
    extractor: ExtractorOption = None,
    embedder: EmbedderOption = None,
) -> None:
    """Write the triple-audit sample (same for every labeller)."""
    config = load_config()
    location = _location(config, dataset, extractor, embedder, "local", False)
    working = location.working_dir(config)
    items = sample_triples(load_graph(working), working, n, config.seed)
    path = audit_dir(working) / "sample.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(i.model_dump_json() + "\n" for i in items), encoding="utf-8")
    console.print(f"Wrote {len(items)} triples to {path}")


@graph_app.command("audit-label")
def audit_label(
    dataset: Annotated[str, typer.Option(help="Dataset whose audit sample to label.")],
    labeller: Annotated[str, typer.Option(help="Your name.")],
    extractor: ExtractorOption = None,
    embedder: EmbedderOption = None,
) -> None:
    """Label triples as correct / partial / wrong against their source text. Type q to stop."""
    config = load_config()
    folder = audit_dir(
        _location(config, dataset, extractor, embedder, "local", False).working_dir(config)
    )
    items = read_models(folder / "sample.jsonl", TripleItem)
    path = folder / f"{labeller}.jsonl"
    done = {label.item_id for label in read_models(path, TripleLabel)}
    choices = {"c": "correct", "p": "partial", "w": "wrong"}
    for index, item in enumerate(items, 1):
        if item.item_id in done:
            continue
        console.rule(f"{index}/{len(items)}")
        console.print(f"[bold]{item.source}[/] → [bold]{item.target}[/]  ({item.keywords})")
        console.print(f"Claim: {item.description}")
        console.print(f"[dim]Source text: {item.evidence[:1200]}[/]")
        key = typer.prompt("verdict [c/p/w/q]").strip().lower()
        while key not in (*choices, "q"):
            key = typer.prompt("please type c, p, w or q").strip().lower()
        if key == "q":
            break
        label = TripleLabel(item_id=item.item_id, labeller=labeller, verdict=choices[key])
        with path.open("a", encoding="utf-8") as handle:
            handle.write(label.model_dump_json() + "\n")


@graph_app.command("audit-kappa")
def audit_kappa(
    dataset: Annotated[str, typer.Option(help="Dataset whose audit to score.")],
    extractor: ExtractorOption = None,
    embedder: EmbedderOption = None,
) -> None:
    """Triple precision per labeller and Cohen's kappa between labellers."""
    config = load_config()
    folder = audit_dir(
        _location(config, dataset, extractor, embedder, "local", False).working_dir(config)
    )
    labels = {
        f.stem: {lab.item_id: lab.verdict for lab in read_models(f, TripleLabel)}
        for f in sorted(folder.glob("*.jsonl"))
        if f.stem != "sample"
    }
    for name, verdicts in labels.items():
        correct = sum(v == "correct" for v in verdicts.values()) + 0.5 * sum(
            v == "partial" for v in verdicts.values()
        )
        console.print(
            f"{name}: triple precision {correct / max(len(verdicts), 1):.1%} (n={len(verdicts)})"
        )
    names = sorted(labels)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = sorted(labels[a].keys() & labels[b].keys())
            if len(shared) >= 2:
                kappa = cohen_kappa([labels[a][k] for k in shared], [labels[b][k] for k in shared])
                mark = "✓" if kappa >= KAPPA_THRESHOLD else "✗"
                console.print(f"kappa({a}, {b}) = {kappa:.3f} on {len(shared)} triples {mark}")
