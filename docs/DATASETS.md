# Dataset access and normalization

Keep downloads in a local data directory. A prepared dataset carries the release identifier entered at preparation time, raw-file SHA-256 values, normalized-file SHA-256 values, seed, and counts. A downloader cache hit verifies the supplied checksum when a checksum file is provided. Use a new destination to preserve earlier input snapshots.

| Dataset | Inputs to `elevate-rag prepare` | Sampling and evidence |
| --- | --- | --- |
| Natural Questions | An official labeled dev JSONL or JSONL gzip file in original format, or an official simplification carrying the same annotations | 500 questions with a short answer or yes/no and an annotated long passage. Pool top-level candidates and any additional annotated candidates. Preserve all available accepted gold candidates and reference alternatives. |
| HotpotQA | Distractor dev JSON or the official distractor validation Parquet | 1,000 records labeled hard. Preserve exact supporting sentence offsets in the joined paragraphs. Pool sampled gold and distractor paragraphs. |
| MuSiQue | `musique_full_v1.0_dev.jsonl` from the official release | Sample 100 unanswerable rows first, then 500 answerable rows with different source IDs. This prevents one variant of the same question from appearing in each split. Preserve supporting paragraph IDs and answer aliases. |
| MultiHop-RAG | `corpus.json MultiHopRAG.json` | Full news corpus and query list. Evidence uses URL and exact local fact offsets where resolvable. Keep evidence text and document-level supervision for the remaining evidence. Null questions require refusal. |
| GraphRAG-Bench | `novel.parquet novel_questions.parquet` or the respective medical files | Full Xiang et al. releases. Qualify question IDs by original source. Keep supporting evidence text and triples even when an exact passage cannot be located. Corpus-wide retrieval is used for internal comparisons. The upstream novel examples group queries by known source, so scores produced under that scope belong in the isolated comparison. |
| RAGRouter-Bench | The directory containing `musique`, `quality`, `ultraDomain_legal` and `graphragBench_medical`, each with `Corpus.json` and `Question.json` | Five-fold TF-IDF/SVM evaluation and leave-one-domain-out validation over the benchmark. The release uses JSON Lines and has untitled legal documents. Stable source IDs identify those documents. Published evidence text is preserved. |
| UltraDomain | User-prepared documents JSONL and questions JSONL | Optional separate experiment. For reference-free questions use `answerable: null`; score summary comparisons pairwise. |

## Input sources and usage terms

- Natural Questions provides labeled development annotations and separate hidden test access rules. The verified input is a pinned mirror of the simplified development set. Check the dataset access terms and Wikipedia attribution alongside the repository license. [Google Research dataset](https://github.com/google-research-datasets/natural-questions), [pinned mirror](https://huggingface.co/datasets/artefactory/natural-questions/tree/749c02daef3e3124c440559868183e6495e41ff5)
- HotpotQA is CC BY-SA 4.0. The loader supports both the original JSON and the authors’ Hugging Face Parquet release. [Project site](https://hotpotqa.github.io/), [official mirror](https://huggingface.co/datasets/hotpotqa/hotpot_qa)
- MuSiQue provides its archive and license through the authors’ repository. The verified input uses the full development split with answerability annotations from a pinned mirror. [MuSiQue release](https://github.com/StonyBrookNLP/musique), [pinned mirror](https://huggingface.co/datasets/voidful/MuSiQue/tree/a7d9f9adf6191604fc67cde318ee1a86fcf7babc)
- MultiHop-RAG lists ODC Attribution on the dataset card. The underlying news articles retain their applicable rights. [Dataset card](https://huggingface.co/datasets/yixuantt/MultiHopRAG)
- GraphRAG-Bench code lists MIT. Check the separate corpus terms, especially NCCN guideline material and the jurisdiction-dependent rights for Gutenberg books, before redistribution. [Benchmark release](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark)
- RAGRouter-Bench documents its four source domains. Check source-specific terms before reusing or redistributing the material. [Authors’ benchmark](https://github.com/ziqiwang0908/RAGRouter-Bench), [dataset card](https://huggingface.co/datasets/Chaplain0908/RAGRouter)

```sh
uv sync --locked --extra graph --extra router --extra faiss --extra datasets
uv run --no-sync elevate-rag download multihop-rag data/input/multihop
uv run --no-sync elevate-rag prepare multihop-rag data/multihop-rag \
  data/input/multihop/corpus.json data/input/multihop/MultiHopRAG.json --version e37cfdd
uv run --no-sync elevate-rag validate-data data/multihop-rag
```

The study split uses exactly floor(`N / 5`) development rows with deterministic allocation across query type, domain, and answerability strata. Official restrictions take precedence. Question records and evidence annotations stay outside the online corpus. Parse errors identify the affected record or file. Unsupported span-level labels remain missing and are kept visible in manifests and metric denominators.
