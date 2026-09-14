# CLI reference

Public entrypoint:

```text
uv run seed
```

Các command chính:

```text
seed doctor --backend local|kaggle
seed source crawl [--sitemap-url URL]
seed build [--leaflets-dir DIR]
seed validate
seed bundle export --output DIR
seed bundle parity --bundle DIR --old-chunks FILE
seed bundle embed --bundle DIR --backend local|kaggle --model MODEL
seed evaluation build [--bundle DIR]
seed embed queries --backend local|kaggle
seed retrieve --run NAME
seed rerank --run NAME --backend local|kaggle --model MODEL
seed metrics --run NAME
```

Global `--json` in envelope máy đọc được, `--debug` bật traceback. `uv run seed COMMAND --help` cho default thực tế.

## Nguồn và build

| Command | Việc |
| --- | --- |
| `seed source crawl [--leaflets-dir DIR] [--sitemap-url URL] [--workers N] [--force] [--dry-run]` | Đọc sitemap index và các sitemap con, giữ trang thuốc dạng `<nhóm>/<slug>`, ghi `html/<nhóm>/<slug>.html`, `urls/all_urls.txt`, `urls/drug_urls.txt` và `manifest.json`; lần đầu cần `--sitemap-url`, các lần sau đọc từ manifest |
| `seed build [--leaflets-dir DIR] [--pdf FILE] [--work-root DIR] [--final-dir DIR]` | Kiểm sha256 từng file HTML theo `data/sources/leaflets/manifest.json`, build trong `data/work/build/in-progress/` rồi publish `data/corpus/rag-final/`; build lỗi giữ ở `data/work/build/failed/latest/` |

## Bundle

| Command | Việc |
| --- | --- |
| `seed bundle export --output DIR [--rag-final-dir DIR] [--glossary FILE] [--mappings FILE] [--force]` | Đọc `rag-final/{sections,blocks,manifest}` và resources, xuất `knowledge-bundle/v1`, validate bằng `read_bundle` trước khi thay `DIR` |
| `seed bundle parity --bundle DIR --old-chunks FILE [--report FILE] [--max-chars N]` | So `chunk_section` của backend với `chunks.jsonl` của build cũ (chunk_text, trang, hydrate_strategy, embedding_text, term_annotations, colloquial_mapping, thứ tự); exit 1 nếu lệch |
| `seed bundle embed --bundle DIR --backend local\|kaggle --model MODEL [--cache FILE] [--work-dir DIR] [--kaggle-account accN] [--dry-run] [--force]` | Gom cặp `(embedding_text_sha256, embedding_text)` duy nhất, embed phần thiếu, ghi `embeddings/<model_slug>.jsonl` và manifest; exit 3 nếu Kaggle chưa xong (chạy lại để resume) |

`--model` của `bundle embed` phải trùng `PHARMA_RETRIEVAL__EMBEDDING__MODEL` của backend, ví dụ `qwen3-embedding:4b-fp16`. Cache nằm ở `data/cache/text_embeddings/<model-slug>.jsonl`; Kaggle dùng dataset input `seed-pipeline-bundle`, stage `corpus-embed` contract version 3.

## Run, retriever và model

`--run NAME` là thư mục `data/evaluation/runs/NAME/`: `run.json` (identity và `origin`), `candidates/`, `rerank/<model>/`, `reports/`. Retrieve, rerank và metrics mở lại cùng thư mục; input khác identity thì lệnh dừng và nêu các trường khác nhau, `--force` thay thế cả run.

`seed retrieve` dựng retrieval stack của backend bằng `build_retrieval_service(settings, embedder=...)`:

- Settings đọc từ `--backend-env-file` (mặc định `../backend/.env`), gồm Postgres, Qdrant, tên và số chiều của model embedding.
- Lệnh ghi đè `retrieval.mode`, `candidate_k`, `prefetch_k`, `rrf_k`, `collections = [--collection]` (mặc định `formulary`), tắt rerank (`protocol = none`) và Langfuse.
- `dense` và `hybrid` dùng vector query có sẵn trong cache của `seed embed queries` (`--query-embeddings`, mặc định `data/cache/query_embeddings/<model-slug>.jsonl` theo `PHARMA_RETRIEVAL__EMBEDDING__MODEL`). Thiếu vector của query nào thì lệnh dừng trước khi retrieve; evaluation không bao giờ embed query qua endpoint. `bm25` không cần vector query.
- Phải có release đã import và publish; release lấy từ query đầu tiên và mọi hit sau phải cùng release.
- Run identity: evaluation path/sha256, collection Qdrant, `embedding_model`, `query_embeddings_sha256`, `retriever`, K, `limit`, `release_id`, `chunker_version`.

| Retriever | Query embeddings | Qdrant | Candidate semantics |
| --- | --- | --- | --- |
| `bm25` | Không dùng | BM25 sparse | `candidate-k` là retrieval limit |
| `dense` | Bắt buộc, từ cache | dense | `candidate-k` là retrieval limit |
| `hybrid` | Bắt buộc, từ cache | dense + BM25 sparse, RRF | `prefetch-k` mỗi nhánh, `candidate-k` sau RRF |

```bash
uv run seed retrieve \
  --run backend-hybrid-qwen4b-p50-k30-rrf2 \
  --retriever hybrid \
  --prefetch-k 50 \
  --candidate-k 30 \
  --rrf-k 2
```

Candidate `chunk_id` là nhãn vị trí `<section_key>:chunk-<ordinal:03d>`, `document_text` là `embedding_text` của chunk (dùng cho rerank).

## Rerank và metrics

`seed rerank` chỉ đọc candidates của run; với Kaggle, stage chỉ load reranker và chấm candidate pairs. Mỗi reranker có đúng một biến thể dưới `rerank/<model>/` của run. `--dry-run` in `target=` và `missing_pairs=N`, số cặp query–chunk chưa có điểm trong `data/cache/rerank_scores/<model>.jsonl`; khi cache đã đủ, rerank không khởi động model.

```bash
uv run seed rerank --run backend-hybrid-qwen4b-p50-k30-rrf2 --backend kaggle --model qwen3-reranker:4b-fp16 --kaggle-account acc2
uv run seed metrics --run backend-hybrid-qwen4b-p50-k30-rrf2 --top-k 30
uv run seed metrics --run backend-hybrid-qwen4b-p50-k30-rrf2 --model qwen3-reranker:4b-fp16 --top-k 30
```

Report nằm ở `reports/baseline/top<K>-window<N>/` và `reports/rerank/<model>/top<K>-window<N>/`; chạy lại cùng cutoff với input khác thì lệnh dừng, `--force` thay report. Metrics gồm Hit@3/5/10/30, MRR, và với `answer_mode=multi_required`: Multi-section Recall@K, Multi-all-hit@K. `--top-k` không được lớn hơn `candidate-k`.

`seed embed queries --backend local|kaggle --model MODEL` tạo cache vector query (`data/cache/query_embeddings/<model-slug>.jsonl`) mà `seed retrieve` dùng cho `dense` và `hybrid`; `--model` phải trùng model embedding của backend. Candidate `document_text` là `embedding_text` của hit, đúng văn bản reranker của benchmark cũ chấm.

## Runtime profiling trên Kaggle

Stage production tự benchmark workload một lần nếu chưa có profile hợp lệ và lưu ở `data/cache/kaggle_profiles/<workload>/<model>.json`; profile mất hiệu lực khi model, runtime, topology hoặc search space đổi. Lỗi benchmark dừng pipeline.

## Exit codes

| Code | Meaning |
| ---: | --- |
| 0 | Stage completed successfully |
| 1 | Runtime or validation failure (bao gồm parity lệch) |
| 2 | Invalid CLI usage |
| 3 | Resumable incomplete stage |
| 130 | Interrupted by the operator |
