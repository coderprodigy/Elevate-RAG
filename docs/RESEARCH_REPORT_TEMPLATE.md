# Hybrid vector and graph retrieval evaluation report

Use [the protocol](EXPERIMENTS.md) and [delivery audit](DELIVERY.md) when preparing the final report.

1. **Introduction and research questions.** Describe the retrieval trade-offs and use the RQ1–RQ6 acceptance mapping.
2. **Background.** Cover lexical, dense and hybrid retrieval, graph retrieval families, learned routing and known evaluation confounders.
3. **System design.** Incorporate the [architecture diagram](ARCHITECTURE.md), exact model and database snapshots, and corpus, retrieval and prompt contracts.
4. **Methodology.** Include dataset versions and sampling counts, frozen comparisons, judge-validation evidence, denominators, costs and statistical procedures.
5. **Results.** Report baseline differences and V0-relative gains, graph effects by query type, router regret and cost, matched database throughput, model factor effects and generator interactions. Give 95% confidence intervals and supported significance tests. Report indexing and query costs.
6. **Analysis.** Add independently annotated failure causes and audit results, examples with original evidence, effect sizes, and threats to validity.
7. **Conclusions.** Answer each research question using measured evidence. Report unresolved comparisons and inconclusive results.
8. **Appendices.** Publish exact configurations, prompts, resource limits, version manifests, per-dataset tables, annotation guidance and contribution records.

The synthetic fixture verifies engineering behavior. Leave numerical headline findings and the research abstract pending until the measured results are frozen.
