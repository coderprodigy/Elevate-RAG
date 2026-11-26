# Datasets

Elevate-RAG evaluates on five datasets plus a router benchmark, with UltraDomain as an optional
extra. This document says where to get each one, its licence, where to put the files, how much
disk it needs and how to check the files are in place.

You do not need any of these for the exit checks. Phases 0–4 run on the bundled synthetic
Tiny_Sample in `fixtures/tiny_sample/`. Download a dataset when you want real numbers for it.

Licence notes below are a summary taken from the upstream pages when this document was written.
They are not legal advice. Check the upstream licence before redistributing any data, and cite every
dataset you report on.

Download URLs change. Where a direct link is given it was checked when this document was written;
if it fails, use the official source page listed for that dataset.

## Contents

- [Expected layout](#expected-layout)
- [Verifying and loading](#verifying-and-loading)
- [Natural Questions](#natural-questions)
- [HotpotQA distractor](#hotpotqa-distractor)
- [MuSiQue](#musique)
- [MultiHop-RAG](#multihop-rag)
- [GraphRAG-Bench](#graphrag-bench)
- [RAGRouter-Bench](#ragrouter-bench)
- [UltraDomain (optional)](#ultradomain-optional)
- [Disk summary](#disk-summary)

## Expected layout

Raw files go under `$ELEVATE_DATA_DIR/raw/<dataset>/` (default `./data/raw/`). Dataset_Loaders
read only these paths and write the Normalised_Corpus and Question_Set to
`$ELEVATE_DATA_DIR/processed/<dataset>/`. File names are case-sensitive.

```
data/
├── raw/
│   ├── nq/
│   │   └── biencoder-nq-dev.json.gz          # or the unpacked biencoder-nq-dev.json
│   ├── hotpotqa/
│   │   └── hotpot_dev_distractor_v1.json
│   ├── musique/
│   │   └── musique_full_v1.0_dev.jsonl       # answerable + unanswerable questions
│   ├── multihop_rag/
│   │   ├── corpus.json
│   │   └── MultiHopRAG.json
│   ├── graphrag_bench/
│   │   ├── Corpus/
│   │   │   ├── medical.json
│   │   │   └── novel.json
│   │   └── Questions/
│   │       ├── medical_questions.json
│   │       └── novel_questions.json
│   ├── ragrouter_bench/
│   │   ├── musique/{Corpus.json,Question.json}
│   │   ├── quality/{Corpus.json,Question.json}
│   │   ├── ultraDomain_legal/{Corpus.json,Question.json}
│   │   └── graphragBench_medical/{Corpus.json,Question.json}
│   └── ultradomain/                          # optional
│       ├── agriculture.jsonl
│       ├── cs.jsonl
│       ├── legal.jsonl
│       └── mix.jsonl
└── processed/
    └── <dataset>/
        ├── corpus.jsonl      # Normalised_Corpus: doc_id, title, text, source
        └── questions.jsonl   # Question_Set: question_id, question, answers, gold_passage_ids,
                              #   query_type, answerable, split
```

The dataset names used on the command line are the directory names above: `nq`, `hotpotqa`,
`musique`, `multihop_rag`, `graphrag_bench`, `ragrouter_bench`, `ultradomain`.

Query_Type is one of `single-hop`, `multi-hop`, `comparison`, `temporal`, `global` or `null` (no
answer in the corpus). Each section below gives the mapping from the source labels.

## Verifying and loading

```bash
uv run elevate verify-setup             # one pass/fail/skipped line per dataset file listed above
uv run elevate data load <dataset>      # convert raw files to processed/<dataset>/ (Phase 1)
```

`verify-setup` reports a missing optional dataset as `skipped`, not `fail`. If a raw file is
missing or has an unexpected schema, `elevate data load` stops with an error naming the file, the
missing field and the section of this document to read.

Loaders use a fixed seed (`ELEVATE_SEED`), so loading the same raw files twice gives identical
output.

## Natural Questions

| | |
|---|---|
| Role | Single-hop factoid baseline; part of the recommended first end-to-end slice |
| What is used | 500 questions from the DPR retriever dev file, with each question's gold (positive) passages plus distractor (hard-negative) passages as the corpus |
| Official source | Natural Questions: <https://ai.google.com/research/NaturalQuestions>. DPR processed files: <https://github.com/facebookresearch/DPR> (see `dpr/data/download_data.py`, resource `data.retriever.nq-dev`) |
| Direct download | <https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz> |
| Licence | Natural Questions is CC BY-SA 3.0. The DPR repository is released under its own licence (marked non-commercial, CC BY-NC 4.0, when this was written; check its `LICENSE`). Academic use is fine under both. |
| File layout | `data/raw/nq/biencoder-nq-dev.json.gz` (or unpacked `.json`) |
| Disk | 256 MB compressed. Keep it compressed if disk is tight; the loader reads either form. |
| Query_Type | every question is `single-hop` |

```bash
mkdir -p data/raw/nq
curl -L -o data/raw/nq/biencoder-nq-dev.json.gz \
  https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz
```

Fields the loader reads per record: `question`, `answers`, `positive_ctxs[]` and
`hard_negative_ctxs[]` (each with `title`, `text`, `passage_id`). The `passage_id` values become
gold passage IDs.

## HotpotQA distractor

| | |
|---|---|
| Role | Multi-hop (bridge) and comparison questions over Wikipedia |
| What is used | 1,000 hard-level questions from the distractor dev set; each question's 10 paragraphs (2 gold, 8 distractors) form the corpus |
| Official source | <https://hotpotqa.github.io/> |
| Direct download | <http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json> (the official link; it did not respond when this document was checked. If it fails, use the Hugging Face mirror below) |
| Mirror | <https://huggingface.co/datasets/hotpotqa/hotpot_qa> (config `distractor`, split `validation`). Convert it to the original JSON format, or download the original file from the official page. |
| Licence | CC BY-SA 4.0 |
| File layout | `data/raw/hotpotqa/hotpot_dev_distractor_v1.json` |
| Disk | about 45 MB |
| Query_Type | `type: bridge` → `multi-hop`; `type: comparison` → `comparison` |

```bash
mkdir -p data/raw/hotpotqa
curl -L -o data/raw/hotpotqa/hotpot_dev_distractor_v1.json \
  http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
```

Fields read: `_id`, `question`, `answer`, `type`, `level`, `context` (list of
`[title, sentences]`) and `supporting_facts` (list of `[title, sentence_index]`). The dev distractor
set is all `level: hard`; the loader still filters on `level`. Gold passage IDs are derived from the
supporting-fact titles.

## MuSiQue

| | |
|---|---|
| Role | 2–4 hop composed questions, plus unanswerable questions for refusal accuracy |
| What is used | 500 answerable and 100 unanswerable dev questions, with each question's paragraphs as the corpus |
| Official source | <https://github.com/StonyBrookNLP/musique> |
| Download | The repository's `download_data.sh` fetches `musique_v1.0.zip` from Google Drive with `gdown`. If `gdown` fails, the script comments give the Drive URL for a manual download. |
| Licence | CC BY 4.0 |
| File layout | `data/raw/musique/musique_full_v1.0_dev.jsonl`. The zip also contains train and test files and the answerable-only `musique_ans_*` files; they are not needed. |
| Disk | not measured for this document. Expect a few hundred MB for the zip and more once unpacked; you can delete everything except the dev file listed above. |
| Query_Type | answerable questions → `multi-hop` (hop count from the `id` prefix, e.g. `2hop__`, `3hop1__`, `4hop3__`, is kept as metadata); `answerable: false` → `null` |

```bash
git clone https://github.com/StonyBrookNLP/musique /tmp/musique
(cd /tmp/musique && bash download_data.sh)
mkdir -p data/raw/musique
cp /tmp/musique/data/musique_full_v1.0_dev.jsonl data/raw/musique/
```

Fields read: `id`, `question`, `answer`, `answer_aliases`, `answerable` and `paragraphs[]` (each
with `idx`, `title`, `paragraph_text`, `is_supporting`). Paragraphs with `is_supporting: true`
become gold passages. The path inside the unpacked zip may differ; find the file with
`find /tmp/musique -name 'musique_full_v1.0_dev.jsonl'`.

## MultiHop-RAG

| | |
|---|---|
| Role | Multi-document news questions: inference, comparison, temporal and null; part of the recommended first end-to-end slice and the Phase 5 graph benchmark |
| What is used | the full query set (2,556 queries) including 301 null queries, over the 609-article news corpus |
| Official source | <https://github.com/yixuantt/MultiHop-RAG> |
| Download | Hugging Face: <https://huggingface.co/datasets/yixuantt/MultiHopRAG> (files `corpus.json`, `MultiHopRAG.json`) |
| Licence | ODC-BY (as tagged on the Hugging Face dataset page). The corpus consists of news articles; respect the original publishers' terms. |
| File layout | `data/raw/multihop_rag/corpus.json`, `data/raw/multihop_rag/MultiHopRAG.json` |
| Disk | about 12 MB |
| Query_Type | `inference_query` → `multi-hop`; `comparison_query` → `comparison`; `temporal_query` → `temporal`; `null_query` → `null` |

```bash
mkdir -p data/raw/multihop_rag
for f in corpus.json MultiHopRAG.json; do
  curl -L -o data/raw/multihop_rag/$f \
    https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/$f
done
```

Corpus fields read: `title`, `body`, `source`, `url`, `published_at`. Query fields read: `query`,
`answer`, `question_type`, `evidence_list[]` (each evidence item names its article by `title` and
`url`, which the loader maps to corpus `doc_id`s).

## GraphRAG-Bench

This is the benchmark from "When to use Graphs in RAG" (arXiv 2506.05690), with novel and medical
subsets and four task levels. A different paper also uses the name GraphRAG-Bench (arXiv
2506.02404); that one is not used here.

| | |
|---|---|
| Role | Graph-oriented questions from fact retrieval to summarisation; evidence recall metric; extractor comparison (medical subset) |
| What is used | the `novel` and `medical` subsets with level 1–4 labels |
| Official source | <https://github.com/GraphRAG-Bench/GraphRAG-Benchmark> |
| Download | Hugging Face: <https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench> (folder `Datasets/`), or the `Datasets/` folder of the GitHub repository |
| Licence | MIT (GitHub repository). The Hugging Face copy has no licence tag; treat it as covered by the repository licence and cite the paper. |
| File layout | `data/raw/graphrag_bench/Corpus/{medical,novel}.json`, `data/raw/graphrag_bench/Questions/{medical,novel}_questions.json` |
| Disk | about 10 MB |
| Query_Type | Level 1 `Fact Retrieval` → `single-hop`; Level 2 `Complex Reasoning` → `multi-hop`; Level 3 `Contextual Summarize` → `global`; Level 4 `Creative Generation` → `global`. The original level is kept as metadata. |

```bash
mkdir -p data/raw/graphrag_bench/Corpus data/raw/graphrag_bench/Questions
base=https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench/resolve/main/Datasets
for s in medical novel; do
  curl -L -o data/raw/graphrag_bench/Corpus/$s.json            $base/Corpus/$s.json
  curl -L -o data/raw/graphrag_bench/Questions/${s}_questions.json $base/Questions/${s}_questions.json
done
```

Corpus records have `corpus_name` and `context`. Question records have `id`, `source`,
`question`, `answer`, `question_type`, `evidence` (list of supporting sentences) and
`evidence_relations`. Evidence is given as text, not passage IDs, so the loader maps each evidence
sentence to the chunks that contain it; GraphRAG-Bench evidence recall uses that mapping.

The GitHub and Hugging Face copies of the question files differ slightly in size. Use one source
for all four files and record which one in your Phase 1 report.

## RAGRouter-Bench

| | |
|---|---|
| Role | Router training and evaluation, including 5-fold cross-validation for R2 |
| What is used | all four subsets: 7,727 queries over 21,460 documents, labelled `factual`, `reasoning` or `summary` |
| Official source | <https://github.com/ziqiwang0908/RAGRouter-Bench> and the paper, arXiv 2602.00296 |
| Download | The repository README says to download the data from a link given in the repository and extract it to `Dataset/RawData/`. No stable direct URL was available when this document was written. Follow the README, then copy `Dataset/RawData/*` into `data/raw/ragrouter_bench/`. |
| Licence | The repository is MIT. The data is derived from MuSiQue, QuALITY, UltraDomain and GraphRAG-Bench, so the upstream licences of those datasets also apply. |
| File layout | `data/raw/ragrouter_bench/<subset>/Corpus.json` and `Question.json` for subsets `musique`, `quality`, `ultraDomain_legal`, `graphragBench_medical` (same structure as the upstream `Dataset/RawData/`) |
| Disk | not measured. The four corpora total about 8.8 million tokens, so expect roughly 40–60 MB of JSON. |
| Query_Type | `factual` → `single-hop`; `reasoning` → `multi-hop`; `summary` → `global` |

Corpus records have `doc_id`, `title`, `text`. Question records have `question_id`, `question`,
`answer`, `query_type`, `supporting_facts` (list of `doc_id`s, used as gold passage IDs).

## UltraDomain (optional)

UltraDomain is the long-document, multi-domain set that the LightRAG paper used for its
comprehensiveness, diversity and empowerment pairwise comparisons. It is optional here; use it for
pairwise LLM-judge comparisons on global questions.

| | |
|---|---|
| Role | Optional pairwise (order-swapped) LLM-judge comparisons on global questions |
| What is used | the four domains used by LightRAG: `agriculture`, `cs`, `legal`, `mix` |
| Official source | <https://huggingface.co/datasets/TommyChien/UltraDomain> (from the MemoRAG project) |
| Licence | Apache-2.0 as tagged on the Hugging Face page. The contexts are drawn from books and other existing datasets (for example NarrativeQA via LongBench); check the original sources before redistributing text. |
| File layout | `data/raw/ultradomain/<domain>.jsonl` |
| Disk | about 280 MB for the four domains (`agriculture` 70 MB, `cs` 95 MB, `legal` 105 MB, `mix` 6 MB). All 20 domains together are about 2.4 GB. |
| Query_Type | every question is `global` |

```bash
mkdir -p data/raw/ultradomain
for d in agriculture cs legal mix; do
  curl -L -o data/raw/ultradomain/$d.jsonl \
    https://huggingface.co/datasets/TommyChien/UltraDomain/resolve/main/$d.jsonl
done
```

Each line has `_id`, `input` (the question), `answers`, `context`, `dataset`, `label` and `length`.

## Disk summary

| Dataset | Raw size | Required for |
|---|---|---|
| Natural Questions | 256 MB (compressed) | E1–E6 runs on NQ; first end-to-end slice |
| HotpotQA distractor | about 45 MB | runs on HotpotQA; E6 |
| MuSiQue | a few hundred MB before cleanup (not measured) | runs on MuSiQue; refusal metrics; E6 |
| MultiHop-RAG | about 12 MB | first end-to-end slice; ablations A1–A4; extractor comparison; Phase 5 graph benchmark |
| GraphRAG-Bench | about 10 MB | evidence recall; extractor comparison (medical); E6 |
| RAGRouter-Bench | about 40–60 MB (estimate) | R2 5-fold cross-validation |
| UltraDomain (4 domains) | about 280 MB | optional pairwise comparisons |

Processed files, chunk stores, vector indexes and graphs take several times the raw size. Graph
indexes for the larger corpora can reach a few GB. Budget 5–10 GB under `data/` and the cache
directory for a full set of experiments.
