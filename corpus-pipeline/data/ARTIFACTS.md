# Data Artifacts

This directory is organized by artifact lifecycle.

## Tracked Inputs

- `raw/duoc-thu-quoc-gia-viet-nam.pdf`: source PDF for the Duoc thu corpus.
- `raw/vietnamese_valid_syllables.json`: dictionary used by text/table cleaning checks.
- `raw/colloquial_mappings.json`: curated An Khang alias and visual-sign mappings.
- `raw/ankhang/all_urls.txt`: An Khang product URL snapshot.
- `raw/ankhang/drug_urls.txt`: filtered An Khang drug URL snapshot.
- `raw/ankhang/html/`: An Khang HTML snapshot used for thesis reproducibility.

## Rebuildable Interim Artifacts

- `interim/text/`: PyMuPDF text extraction and cleaned Markdown.
- `interim/rag/`: intermediate sections and chunks before canonical merge.
- `interim/docling/`: Docling table extraction, curation outputs, and curated/manual override files.
- `interim/canonical/`: canonical block assembly and audit artifacts.
- `interim/ankhang_markdown/`: parsed An Khang Markdown used to build final brand-page sections.

Generated interim files should not be added to git by default. Curated/manual override files that are needed for deterministic rebuilds may be tracked.

## Tracked Processed Contracts

- `processed/rag-final/`: final RAG corpus contract used by validation, metadata build, Postgres import, Qdrant ingest, and retrieval evaluation.
- `processed/evaluation/patient_queries.json`: official patient-style seed queries.
- `processed/evaluation/section_retrieval_eval.csv`: official retrieval evaluation dataset.

These files are tracked so thesis runs can be reproduced without rebuilding every pipeline stage.

## Local-Only Outputs

- `runs/retrieval_eval/`: retrieval experiment JSONL, CSV, JSON, and Markdown outputs.
- `cache/query_embeddings/`: local query embedding cache created by dense or hybrid `cli.run_retrieval_eval` runs.
- `cache/vector_embeddings/`: local corpus chunk embedding resumability cache created by real `cli.ingest_vectors` runs.

Cache directories may be absent in a fresh checkout or after cleanup. The relevant CLI command recreates them on demand.

Local-only outputs are ignored by git. Regenerate them with the relevant CLI command when needed.

## Rules For New Code

- Use `src/config/paths.py` for default data paths.
- Put official datasets under `processed/`.
- Put experiment outputs under `runs/`.
- Put resumability caches under `cache/`.
- Do not add Python bytecode or `__pycache__` directories under `data/`.
