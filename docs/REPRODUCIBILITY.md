# Reproducing the audit

Raw evidence paths below refer to local workspace files. See the [checkpoint evidence](evidence/README.md) for the published summaries.

Run these commands from the Atlas checkout on Python 3.12 with uv. The main lock is `uv.lock`. Raw inputs, caches, virtual environments, databases and reports belong in the local ignored directories. The Phase 2 test host was macOS arm64 on Apple M4 with 16 GB RAM. Environment snapshot (local evidence, `artifacts/completion/environment.json`)

## Locked offline quick start

```sh
uv sync --locked --extra graph --extra router --extra faiss
uv lock --check
uv run --no-sync elevate-rag prepare-fixture data/synthetic
uv run --no-sync elevate-rag index data/synthetic --graph
uv run --no-sync elevate-rag validate-index data/synthetic --mode mix
uv run --no-sync elevate-rag run data/synthetic --mode mix --router R4 \
  --split all --run-id offline-r4
make check
```

Run the same command and run ID again to verify resume. The clean-install phase used a fresh named virtual environment and completed a full 20-question V4 run. The default tests skip opt-in service checks. Use the storage extra for the test database drivers and dedicate local endpoints, project names and volumes. Clean install log (local evidence, `artifacts/phase2/clean-install.log`), clean fixture run (local evidence, `artifacts/phase2/clean-results/synthetic/phase2-clean-v4/report.md`), [database setup](EXPERIMENTS.md)

## Benchmark inputs and verification

```sh
uv sync --locked --extra graph --extra router --extra faiss --extra datasets
for name in natural-questions hotpotqa musique graphrag-novel graphrag-medical ragrouter-bench; do
  uv run --no-sync elevate-rag download "$name" "data/input/$name"
done
uv run --no-sync elevate-rag download multihop-rag data/input/multihop

uv run --no-sync elevate-rag prepare natural-questions data/natural-questions \
  data/input/natural-questions/simplified-nq-dev.jsonl.gz \
  --version simplified-dev-artefactory-749c02daef3e3124c440559868183e6495e41ff5
uv run --no-sync elevate-rag prepare hotpotqa data/hotpotqa \
  data/input/hotpotqa/validation-00000-of-00001.parquet --version distractor-v1-author-parquet
uv run --no-sync elevate-rag prepare musique data/musique \
  data/input/musique/musique_full_v1.0_dev.jsonl \
  --version full-v1.0-dev-voidful-a7d9f9adf6191604fc67cde318ee1a86fcf7babc
uv run --no-sync elevate-rag prepare multihop-rag data/multihop-rag \
  data/input/multihop/corpus.json data/input/multihop/MultiHopRAG.json --version e37cfdd
for domain in novel medical; do
  uv run --no-sync elevate-rag prepare "graphrag-$domain" "data/graphrag-$domain" \
    "data/input/graphrag-$domain/$domain.parquet" \
    "data/input/graphrag-$domain/${domain}_questions.parquet" --version upstream-2026-11-26
done
uv run --no-sync elevate-rag prepare ragrouter-bench data/ragrouter-bench \
  data/input/ragrouter-bench --version chaplain0908-abff378d9202caba636aa4d5bb9243db8103f70f
uv run --no-sync python scripts/audit_datasets.py
uv run --no-sync elevate-rag router-cv data/ragrouter-bench
uv run --no-sync elevate-rag router-cv data/ragrouter-bench --leave-domain-out
```

The downloader uses revision URLs recorded in the audit artifact. Preparation produces raw and processed SHA-256 hashes, labels, split sizes and evidence mappings. The audit independently rebuilds each dataset from its matching raw files, validates the evidence bounds and compares normalized bytes. Use `--refresh NAME` to migrate a legacy split inside this workspace and preserve its original manifest and partition files before rewriting. [Data access](DATASETS.md), dataset audit (local evidence, `artifacts/phase2/dataset-audit.json`)

## Native fixture runs

Create the synthetic data first. Each framework runs in a separate Python environment with loopback chat and embedding fixtures and a distinct output directory. A fresh Microsoft environment takes roughly 730 MiB when unpacked. HippoRAG took roughly 817 MiB installed. Installation also needs working space for downloads and extraction. The commands can be run sequentially to reduce disk pressure.

```sh
mkdir -p artifacts/native
uv venv --python 3.12 artifacts/native/ms-venv
uv pip install --python artifacts/native/ms-venv/bin/python 'graphrag==3.3.0'
export TIKTOKEN_CACHE_DIR="$PWD/artifacts/native/tokenizers"
artifacts/native/ms-venv/bin/python -c 'import tiktoken; tiktoken.get_encoding("o200k_base"); tiktoken.get_encoding("cl100k_base")'
uv run --no-sync python scripts/native_fixture.py microsoft-graphrag 3.3.0 \
  artifacts/native/ms-venv/bin/python artifacts/native/microsoft

curl -fsSL 'https://codeload.github.com/OSU-NLP-Group/HippoRAG/tar.gz/d5c8329422e0a0b834a15874545cb6a74b4f9b26' \
  -o artifacts/native/hipporag-2.0.0a5.tar.gz &&
  printf '%s  %s\n' '0aa7bad1b5753ce6991826ac91a6550aae66d2a5c0901679b5272c4742b82b2a' \
    artifacts/native/hipporag-2.0.0a5.tar.gz | shasum -a 256 -c - &&
  uv venv --python 3.12 artifacts/native/hippo-venv &&
  uv pip install --python artifacts/native/hippo-venv/bin/python artifacts/native/hipporag-2.0.0a5.tar.gz &&
  uv run --no-sync python scripts/native_fixture.py hipporag2 2.0.0a5 \
    artifacts/native/hippo-venv/bin/python artifacts/native/hipporag --port 28711
```

The first tokenizer invocation fetches public tokenizer assets. The fixture endpoints stay on loopback and use no paid credentials. The native execution manifest pins wrapper bytes, export bytes and runtime settings. Use a new output directory for changed code or framework versions. The historical freeze files capture transitive versions used in the audit. The HippoRAG freeze's source entry is a local install path. Recreate it with the pinned archive above. Microsoft dependencies (local evidence, `artifacts/phase2/ms-dependencies.txt`), HippoRAG dependencies (local evidence, `artifacts/phase2/hippo-dependencies.txt`), HippoRAG source checksum (local evidence, `artifacts/phase2/hippo-archive.sha256`)

## Audit and run artifacts

```sh
uv run --no-sync pytest --cov=elevate_rag --cov-report=term-missing \
  --cov-report=json:artifacts/phase2/coverage.json
uv run --no-sync python scripts/profile_retrieval_metrics.py data/natural-questions artifacts/phase2/performance
uv run --no-sync python scripts/estimate_study.py
```

Preserve the exact data, configuration, dependency versions and checkpoint directory when resuming. Research outcomes incorporate configuration fingerprints, data hashes and package source versions. Keep secrets in environment variables. [Ignore rules](../.gitignore), [credential names](../.env.example), [research runbook](RESEARCH_EXECUTION_PLAN.md)
