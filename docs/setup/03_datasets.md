# 03 — Datasets

**Everything downloads automatically, so there is no manual step.** A fresh clone needs only:

```bash
uv run hgrag data download     # ~400 MB, free; skips files already present
uv run hgrag data build        # data/processed/<name>/ (about 30 s)
uv run hgrag data card         # summary table + reports/data_cards.md
```

The frozen `splits.json` files are committed to git, so every teammate gets **identical** dev/test
splits. A build fails on purpose if the sampled questions no longer match the frozen split.

```
data/
├── raw/<source>/                 downloads (git-ignored)
└── processed/<dataset>/
    ├── corpus.jsonl              Document: doc_id, title, text, source, meta     (git-ignored)
    ├── queries.jsonl             Query: qid, question, answers, gold_doc_ids, gold_evidence,
    │                                    query_type, native_type, hop_count, answerable, split
    ├── datacard.json                                                               (git-ignored)
    └── splits.json               FROZEN dev/test assignment                        (tracked)
```

## Sources (format checked on 2026-11-24)

| Dataset | Source | Sample | Notes |
|---|---|---|---|
| `nq` | DPR `biencoder-nq-dev.json.gz` (dl.fbaipublicfiles.com) | 500 questions | Corpus = gold passages + 10 DPR hard negatives per question |
| `hotpotqa` | HF `hotpotqa/hotpot_qa`, distractor/validation | 1,000 | All dev questions are level "hard"; paragraphs pooled into one corpus |
| `musique` | HF mirror `bdsaglam/musique` (`musique_full_v1.0_dev.jsonl`) | 500 answerable + 100 unanswerable | Unanswerable questions kept only if their missing evidence stays out of the pooled corpus |
| `multihop_rag` | HF `yixuantt/MultiHopRAG` | all 2,556 | Gold = article title + fact sentence |
| `graphrag_bench_novel` / `_medical` | HF `GraphRAG-Bench/GraphRAG-Bench` | all | Whole books / one guideline text; gold evidence is sentence text |
| `ragrouter_bench` | HF `Chaplain0908/RAGRouter` | all, minus overlaps | Router only. Questions also in our datasets are dropped (all of its medical domain) |
| UltraDomain (optional) | HF `TommyChien/UltraDomain` | — | Not wired in yet (Could tier) |

The official MuSiQue release is on Google Drive (github.com/StonyBrookNLP/musique). The Hugging Face
mirror's `ans` dev file is byte-identical in size to a second independent mirror. If you want to use
the official files instead, put `musique_full_v1.0_dev.jsonl` in `data/raw/musique/` and the downloader
will skip the mirror.

## Hugging Face token

Downloads work anonymously. A `HF_TOKEN` in `.env` only raises rate limits. On your Mac, a token saved
by `huggingface-cli login` in March 2024 (`~/.cache/huggingface/token`) is no longer valid and makes
the HF library fail with HTTP 401. Our downloader ignores that file. To clean it up for other tools,
run `huggingface-cli logout` or log in again with a fresh token.

## Licences

Recorded per dataset in `configs/datasets.yaml` and in the data cards: HotpotQA CC BY-SA 4.0,
MuSiQue CC BY 4.0, NQ CC BY-SA 3.0, RAGRouter-Bench MIT. **MultiHop-RAG and GraphRAG-Bench: check the
source pages before redistributing anything.** Raw data is never committed.
