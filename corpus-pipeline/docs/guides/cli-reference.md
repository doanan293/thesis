# CLI reference

The public entrypoint is `uv run corpus`. Commands are stage-oriented and use the same flags across local and Kaggle execution.

```text
corpus doctor
corpus build
corpus validate
corpus source crawl
corpus source extract-tables
corpus source curate-tables
corpus evaluation build
corpus embed chunks --backend local|kaggle
corpus embed queries --backend local|kaggle
corpus vectors upload
corpus retrieve --run NAME
corpus rerank --run NAME --backend local|kaggle
corpus metrics --run NAME
```

Global `--json` emits a machine-readable envelope and `--debug` enables tracebacks.
Use `--help` on any command for defaults. Model stages share `--backend`,
`--model`, `--force`, `--dry-run`, `--budget-seconds`, and
`--request-timeout-seconds` where applicable. `local` and `kaggle` change only
where model work executes; artifact contracts remain identical.

Artifacts are published as bundles with manifests. Retrieval, reranking, and
metrics share a named workspace under `data/runs/retrieval_eval/` through
`--run NAME`.

Exit codes:

| Code | Meaning |
| ---: | --- |
| 0 | Stage completed successfully |
| 1 | Runtime or validation failure |
| 2 | Invalid CLI usage |
| 3 | Resumable incomplete stage |
| 130 | Interrupted by the operator |
