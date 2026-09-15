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
seed bundle embed --bundle DIR --backend local|kaggle --model MODEL
seed evaluation build [--bundle DIR]
seed embed queries --backend local|kaggle
seed retrieve --run NAME
seed rerank --run NAME --backend local|kaggle --model MODEL [--kaggle-account accN|auto] [--max-runs N]
seed rerank --run NAME --backend local --benchmark --model MODEL
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

Với `--backend kaggle`, job chạy thành nhiều phiên GPU. `--kaggle-account auto` đọc `kaggle quota -v` của mọi profile trước mỗi phiên và chọn tài khoản còn nhiều giờ nhất (bằng nhau thì `accN` nhỏ hơn) mà không bị lệnh khác khoá; `--kaggle-account accN` giữ một tài khoản. Ngân sách phiên = min(`--budget-seconds`, quota còn lại − 0,5 giờ), tối thiểu 1 giờ nên `--budget-seconds` phải ≥ 3600. Lệnh dừng khi đủ cặp (`stop=complete`), hết quota (`stop=quota-exhausted`, exit 3, in bảng quota với `refresh_at`), một phiên không thêm cặp (`stop=no-progress`) hoặc đạt `--max-runs N` (`stop=max-runs`). Điểm của mọi phiên được gộp vào cache ở máy trước khi publish checkpoint, và cache đủ cặp thì lệnh ghi biến thể mà không gọi Kaggle. `--force` chấm lại từ đầu: xoá các cặp của run khỏi cache ở máy, phiên đầu bỏ qua checkpoint, kernel cũ và publish lại dataset, các phiên sau chỉ tiếp tục từ điểm của lần chạy này. Mỗi lệnh ghi log có timestamp vào `data/work/logs/rerank/<model>.log`.

Rerank chỉ gọi `POST /v1/rerank` với file GGUF bản convert classifier; mỗi request là một câu hỏi cùng các ứng viên chưa có điểm, ở local lẫn Kaggle. `--backend local` dùng service `llama-reranker` của `../compose.yaml`: nếu `data/cache/local_profiles/rerank/<model>.json` khớp model, CPU (`/proc/cpuinfo`) và image llama.cpp của compose thì service được dựng theo profile, không thì theo mặc định của compose.

`--backend local --benchmark` đo từng mức của search space CPU (`-np` 16; `-ub` 4096/8192/16384, dùng chung cho `-c` và `-b`; `--threads` 8/12). Mỗi mức dựng lại `llama-reranker`, chạy một nhóm khởi động không tính giờ rồi 6 nhóm câu hỏi × 15 ứng viên chọn phân tầng theo tổng số ký tự. Mức có điểm lệch mức hợp lệ đầu tiên quá `1e-3` bị loại; lệnh chọn p95 thấp nhất (bằng nhau thì số cặp/giây cao hơn), lưu profile và in `selected=`, `latency_p95_seconds=`, `env=`. Chép `env=` vào `.env` ở gốc repo và đặt `RERANK_TIMEOUT_SECONDS` ít nhất gấp đôi `latency_p95_seconds`.

```bash
uv run seed rerank --run hybrid-qwen4b-p50-k30-rrf2 --backend local --benchmark --model qwen3-reranker:4b-fp16
```

```bash
uv run seed rerank --run backend-hybrid-qwen4b-p50-k30-rrf2 --backend kaggle --model qwen3-reranker:4b-fp16 --kaggle-account auto
uv run seed metrics --run backend-hybrid-qwen4b-p50-k30-rrf2 --top-k 30
uv run seed metrics --run backend-hybrid-qwen4b-p50-k30-rrf2 --model qwen3-reranker:4b-fp16 --top-k 30
```

Report nằm ở `reports/baseline/top<K>-window<N>/` và `reports/rerank/<model>/top<K>-window<N>/`; chạy lại cùng cutoff với input khác thì lệnh dừng, `--force` thay report. Metrics gồm Hit@3/5/10/30, MRR, và với `answer_mode=multi_required`: Multi-section Recall@K, Multi-all-hit@K. `--top-k` không được lớn hơn `candidate-k`.

`seed embed queries --backend local|kaggle --model MODEL` tạo cache vector query (`data/cache/query_embeddings/<model-slug>.jsonl`) mà `seed retrieve` dùng cho `dense` và `hybrid`; `--model` phải trùng model embedding của backend. Candidate `document_text` là `embedding_text` của hit, đúng văn bản reranker của benchmark cũ chấm.

## Runtime profiling trên Kaggle

Stage production tự benchmark một lần nếu chưa có profile hợp lệ và lưu ở `data/cache/kaggle_profiles/<workload>/<model>.json`; profile mất hiệu lực khi model, runtime, topology hoặc search space đổi. Mỗi mức là một cấu hình server đầy đủ (`-np`, `-ub`, số tài liệu mỗi request, số request đồng thời) và server khởi động lại khi đổi mức. Reranker chạy `--reranking --kv-unified -np N -c UB -b UB -ub UB`; tải đo là 32 nhóm câu hỏi × 30 ứng viên chọn phân tầng theo tổng số ký tự, cộng một nhóm khởi động không tính giờ. Mức nào có điểm lệch mức hợp lệ đầu tiên quá `1e-3` bị ghi `invalid` (`score_mismatch`); stage chọn mức có số cặp/giây cao nhất. Mọi mức đều lỗi thì stage dừng và in trạng thái cùng đuôi log server của từng mức.

## Archive dữ liệu

Mọi file dưới `data/` mà Git bỏ qua (trừ `data/work/`) được lưu trong dataset Kaggle private `<owner>/seed-pipeline-data`, `<owner>` là username của profile `--kaggle-account`. `push` đóng gói thành các phần `seed-pipeline-data.tar.zst.part-NNNN` (tối đa 1.900 MiB mỗi phần) cùng `archive-manifest.json` ghi sha256 từng phần và từng file, dựng ở `data/work/archive/`, rồi tạo dataset hoặc version mới.

```bash
uv run seed data push --kaggle-account acc1 --message "Rebuild evaluation runs"
uv run seed data pull --kaggle-account acc1
```

`pull` tải lại các phần còn thiếu hoặc lệch checksum, kiểm sha256 từng phần và từng file, bỏ qua file đã giống hệt. File đang có mà khác archive thì lệnh dừng và liệt kê; thêm `--force` để ghi đè.

## Exit codes

| Code | Meaning |
| ---: | --- |
| 0 | Stage completed successfully |
| 1 | Runtime or validation failure |
| 2 | Invalid CLI usage |
| 3 | Resumable incomplete stage |
| 130 | Interrupted by the operator |
