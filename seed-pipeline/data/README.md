# Data layout

```text
data/
├── sources/      original inputs: the formulary PDF, leaflets/ (manifest.json, urls/,
│                 html/<category>/<slug>.html), curation/, colloquial_mappings.json,
│                 term_glossary.json, vietnamese_valid_syllables.json
├── corpus/       rag-final/ (output of `seed build`) and formulary/ (the knowledge bundle
│                 that `pharma-agent corpus import` loads)
├── evaluation/   gold/ (evaluation set) and runs/<run>/ (run.json, candidates/,
│                 rerank/<model>/, reports/)
├── cache/        text_embeddings/<model>.jsonl, query_embeddings/<model>.jsonl,
│                 rerank_scores/<model>.jsonl, kaggle_profiles/<workload>/<model>.json
└── work/         scratch space: build workspace, locks, bundle-embed input, archive staging
```

`<model>` is the model slug from the catalog, for example `qwen3_embedding_4b_fp16`. File and folder names never contain hashes; digests and identities live inside `manifest.json` files and cache records.

## What Git tracks

Git keeps only small files that are maintained by hand or quoted in the thesis report:

- everything in `sources/` except the PDF and the leaflet HTML;
- `manifest.json` and `validation_report.json` of the corpus artifacts;
- `run.json`, `manifest.json` and `report.md` of evaluation runs.

Everything else under `data/`, except `work/`, lives in a private Kaggle dataset and is stored and restored with `seed data push` and `seed data pull`, which check the sha256 of every part and every file:

```bash
uv run seed data push --kaggle-account acc1 --message "Rebuild evaluation runs"
uv run seed data pull --kaggle-account acc1
```

The first push after a fresh setup needs the owner's confirmation; pull refuses to overwrite files that differ from the archive unless `--force` is given.

## Leaflet source

`sources/leaflets/manifest.json` (schema `leaflet-source-v1`) is the only record of where the leaflet pages came from: the sitemap URL, the crawl date, the sha256 of the URL list, and the path, size, sha256 and source URL of every HTML file. It stays inside seed-pipeline; the knowledge bundle names leaflets only by `leaflet:<category>:<slug>`. `seed build` refuses to run when a file no longer matches the manifest.

`heavy/` is the pre-migration folder; the one-off data migration moves its contents into the layout above and then deletes it.
