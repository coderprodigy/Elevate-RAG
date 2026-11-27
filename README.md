# Elevate-RAG

Elevate-RAG compares hybrid lexical/dense retrieval, LightRAG graph retrieval, and query routing over a shared chunk catalog. Dense vector baselines and LightRAG read one physical passage index. It records the evidence passed to the generator, per-question outcomes, spend estimates, and paired statistics.

The offline workflow uses a disclosed synthetic corpus. See the [Phase 2 delivery](docs/FINAL_DELIVERY.md) for verification results, research gates, and open work.

## Local quick start

Python 3.12 and uv are required. Run from the checkout root.

```sh
uv sync --locked --extra graph --extra router --extra faiss
uv run --no-sync elevate-rag environment
uv run --no-sync elevate-rag prepare-fixture data/synthetic
uv run --no-sync elevate-rag run data/synthetic --mode V4 --split all --run-id local-demo
uv run --no-sync elevate-rag index data/synthetic --graph
uv run --no-sync elevate-rag inspect-graph data/synthetic artifacts/synthetic-audit
uv run --no-sync elevate-rag answer data/synthetic 'Which beacon belongs to Station Aster?' --mode mix
make check
```

Run outputs appear under `results/synthetic/<run-id>`. Use the same run ID and identical arguments to resume. Add `--retry-failed` to run only pending and failed questions. A changed manifest requires a new run ID.

## Research workflows

1. Prepare upstream data using [dataset access and normalization](docs/DATASETS.md). Keep raw and normalized files under the ignored `data/` directory.
2. Review the [protocol and model setup](docs/EXPERIMENTS.md). Create a local copy of [the reference candidate](configs/reference.toml) with verified model serving, resource limits, and the approved spending cap.
3. Build the shared vector index. Run the representative 5% graph pilot before any full paid indexing. Inspect graph coverage and export the triple audit before collecting held-out graph answers.
4. Use `tune-v4` on development evidence to freeze V4 and calibrate R4 with `calibrate-router`. Use `--split heldout` for final measurements.
5. Compare complete runs with `compare` and generate reports with `report`. Use `matrix` to inspect E1–E6 and `suite` to validate configured arms.

The default configuration uses the offline synthetic providers. Real datasets require a real configuration and exact tokenizer revisions. [Requirements](docs/REQUIREMENTS.md), [architecture](docs/ARCHITECTURE.md), [traceability](docs/REQUIREMENTS_TRACEABILITY.md).
