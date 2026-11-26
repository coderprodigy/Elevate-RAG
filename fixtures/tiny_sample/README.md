# Tiny_Sample fixture

A synthetic corpus and question set, bundled so the Phase 0 to 4 exit checks run offline without
downloaded datasets (Requirement 5.7). It holds 20 questions, within the Tiny_Sample limit set by
Requirement 7.4.

All entities are invented (tidal energy around the fictional Varn Strait), so a closed-book model
cannot answer from memory. V0 should score near zero, which is the expected contamination-free
result.

## Files

| File | Content |
|---|---|
| `corpus.jsonl` | 17 Normalised_Corpus records `{doc_id, title, text, source}`. Docs 01 to 12 support questions; docs 13 to 17 are distractors, some of them near misses (for example a "Kestrel Ridge" trail restoration cost). |
| `questions.jsonl` | 20 Question_Set records `{question_id, question, answers, gold_passage_ids, query_type, answerable, split}`. |
| `fake_llm/extraction.jsonl` | One scripted extraction response per document. |
| `fake_llm/generation.jsonl` | Scripted generator responses per `(question_id, variant)`. |
| `fake_llm/judge.jsonl` | Scripted LLM_Judge verdicts for every scripted generator answer. |
| `fake_llm/malformed.jsonl` | Deliberately broken responses for error-path tests (Requirements 3.6, 10.4, 13.6, 15.5). |
| `manifest.json` | Counts, formats and constants. |
| `build.py` | Source of truth. Regenerates every file above deterministically. |
| `validate.py` | Integrity checks: schemas, gold IDs, Query_Type coverage, extraction format, judge labels. |

Do not edit the generated files by hand. Edit `build.py`, then run:

```
python fixtures/tiny_sample/build.py
python fixtures/tiny_sample/validate.py
```

## Questions

| Query_Type | IDs | Notes |
|---|---|---|
| single-hop | q01 to q04 | One gold document each. |
| multi-hop | q05 to q08 | 2 or 3 gold documents. Consecutive gold documents share an extracted entity, so a graph path exists. No single gold document answers the question alone; for example doc 02 (Mirela Danthorpe) does not name the Halvren Institute. |
| comparison | q09 to q11 | Two gold documents, one per compared item. |
| temporal | q12 to q14 | Date arithmetic or interval overlap. |
| global | q15 to q17 | Corpus-level summaries; the single reference answer is a summary, so EM/F1 are low by design and the judge is the primary metric. |
| null | q18 to q20 | `answerable: false`, `answers: []`, `gold_passage_ids: []`. The expected answer is the Refusal_String `insufficient information`. |

`gold_passage_ids` are `doc_id`s. Each document is at most 120 words, so it fits in one chunk at every
configured chunk size (256, 512, 1,024 tokens), and gold chunk IDs map 1:1 to gold documents.

`split` is `null` so the Splitter assigns dev/test itself (Requirement 7.1). Callers that need the
fixture as a ready-made dev Tiny_Sample can treat every question as dev.

## Scripted fake LLM responses

### Extraction (`role: extraction`)

Each record holds `doc_id`, `match` (the first sentence of the document; a fake can select the
record when the prompt contains this string, since chunk IDs are produced by the Chunker),
`prompt_version`, `response`, `entities` and `relations`.

`response` uses LightRAG 1.5.x delimited rows, one row per line, ending with the completion marker:

```
entity<|#|>Kestrel Tidal Array<|#|>Artifact<|#|>Tidal power station in the Varn Strait, ...
relation<|#|>Kestrel Tidal Array<|#|>Varn Strait<|#|>located in<|#|>The Kestrel Tidal Array is in the Varn Strait.
<|COMPLETE|>
```

LightRAG's row format has no weight field. `relations[].weight` (0 to 1) carries the weight that
Requirement 15.1 requires; the `relation` field holds the LightRAG keywords. `entities` and
`relations` are the same rows in parsed form, so tests can assert parser output against them.

Entity names are title case and consistent across documents so cross-document links form multi-hop
paths. One duplicate is intentional: doc 08 extracts `Varn strait` (lower-case s), which the
Entity_Resolver should merge into `Varn Strait` through identical normalised names (Requirement
16.1).

### Generation (`role: generator`)

Each record holds `question_id`, `variant`, `response_template`, `answer` (citations removed) and
`cited_doc_ids`. Variants:

- `rag`: the correct grounded answer (refusal for null questions). Present for every question.
- `refusal`: the Refusal_String, for answerable questions (judged `wrong`, a false refusal).
- `closed_book`: the V0 answer. Refusal for most questions, confident wrong guesses for q01, q02
  and q10.
- `partial`: an incomplete answer for the global questions (judged `partial`).

`response_template` marks citations as `[[doc_id]]`, because passage numbers depend on the order of
the numbered context. Render a template before returning it from a fake: replace each `[[doc_id]]`
with `[n]`, where `n` is the number of the first context passage whose chunk belongs to that
document, and drop the marker when that document is not in the context. A suggested fake policy: if
none of `cited_doc_ids` is in the context, return the Refusal_String. Under that policy accuracy
depends on retrieval recall, which keeps the exit checks meaningful.

### Judging (`role: judge`)

Each record holds `question_id`, `variant`, `prediction` (equal to the generation `answer`) and
`response`, a JSON string `{"label": "correct|partial|wrong", "rationale": "..."}`. Match on
`question_id` plus the normalised prediction with citation markers such as `[1]` removed. For
predictions with no scripted verdict, a fake can fall back to normalised exact match against
`answers` (or the Refusal_String for null questions). Pairwise comparison verdicts are not scripted.

### Malformed examples

`malformed.jsonl` holds one truncated extraction (wrong delimiter, no completion marker), two
invalid judge outputs (non-JSON text and an out-of-set label) and one generation that cites a passage
number absent from any context.
