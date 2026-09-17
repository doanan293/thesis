# CLI reference

Public entrypoint:

```text
uv run pharma-lab
```

Các command chính:

```text
pharma-lab doctor --backend local|kaggle
pharma-lab source crawl [--sitemap-url URL]
pharma-lab build [--leaflets-dir DIR]
pharma-lab validate
pharma-lab bundle export --output DIR
pharma-lab bundle embed --bundle DIR --backend local|kaggle --model MODEL
pharma-lab evaluation build [--bundle DIR]
pharma-lab evaluation judgments [--evaluation FILE] [--bundle DIR]
pharma-lab embed queries --backend local|kaggle
pharma-lab retrieve --run NAME [--limit N | --sample N [--sample-seed S]]
pharma-lab rerank --run NAME --backend local|kaggle --model MODEL [--kaggle-account accN|auto] [--max-runs N] [--recover-kernel OWNER/SLUG]
pharma-lab rerank --run NAME --backend local --benchmark --model MODEL
pharma-lab metrics --run NAME
pharma-lab metrics compare --run NAME --baseline MODEL --candidate MODEL
pharma-lab e2e golden sample|resample|check|build
pharma-lab e2e run --run NAME --config CONFIG [--limit N] [--deadline-seconds S] [--retry-errors]
pharma-lab e2e judge --run NAME --config CONFIG [--judge-model MODEL] [--force]
pharma-lab e2e calibration export|score --run NAME
pharma-lab e2e report --run NAME
pharma-lab e2e rerank-server --model MODEL --hours H --kaggle-account accN|auto
```

Global `--json` in envelope máy đọc được, `--debug` bật traceback. `uv run pharma-lab COMMAND --help` cho default thực tế.

## Nguồn và build

| Command | Việc |
| --- | --- |
| `pharma-lab source crawl [--leaflets-dir DIR] [--sitemap-url URL] [--workers N] [--force] [--dry-run]` | Đọc sitemap index và các sitemap con, giữ trang thuốc dạng `<nhóm>/<slug>`, ghi `html/<nhóm>/<slug>.html`, `urls/all_urls.txt`, `urls/drug_urls.txt` và `manifest.json`; lần đầu cần `--sitemap-url`, các lần sau đọc từ manifest |
| `pharma-lab build [--leaflets-dir DIR] [--pdf FILE] [--work-root DIR] [--final-dir DIR]` | Kiểm sha256 từng file HTML theo `data/sources/leaflets/manifest.json`, build trong `data/work/build/in-progress/` rồi publish `data/corpus/rag-final/`; build lỗi giữ ở `data/work/build/failed/latest/` |

## Bundle

| Command | Việc |
| --- | --- |
| `pharma-lab bundle export --output DIR [--rag-final-dir DIR] [--glossary FILE] [--mappings FILE] [--force]` | Đọc `rag-final/{sections,blocks,manifest}` và resources, xuất `knowledge-bundle/v1`, validate bằng `read_bundle` trước khi thay `DIR` |
| `pharma-lab bundle embed --bundle DIR --backend local\|kaggle --model MODEL [--cache FILE] [--work-dir DIR] [--kaggle-account accN] [--dry-run] [--force]` | Gom cặp `(embedding_text_sha256, embedding_text)` duy nhất, embed phần thiếu, ghi `embeddings/<model_slug>.jsonl` và manifest; exit 3 nếu Kaggle chưa xong (chạy lại để resume) |

`--model` của `bundle embed` phải trùng `PHARMA_RETRIEVAL__EMBEDDING__MODEL` của backend, ví dụ `qwen3-embedding:4b-fp16`. Cache nằm ở `data/cache/text_embeddings/<model-slug>.jsonl`; Kaggle dùng dataset input `seed-pipeline-bundle`, stage `corpus-embed` contract version 3.

## Run, retriever và model

`--run NAME` là thư mục `data/evaluation/runs/NAME/`: `run.json` (identity và `origin`), `candidates/`, `rerank/<model>/`, `reports/`. Retrieve, rerank và metrics mở lại cùng thư mục; input khác identity thì lệnh dừng và nêu các trường khác nhau, `--force` thay thế cả run.

`pharma-lab retrieve` dựng retrieval stack của backend bằng `build_retrieval_service(settings, embedder=...)`:

- Settings đọc từ `--backend-env-file` (mặc định `../backend/.env`), gồm Postgres, Qdrant, tên và số chiều của model embedding.
- Lệnh ghi đè `retrieval.mode`, `candidate_k`, `prefetch_k`, `rrf_k`, `collections = [--collection]` (mặc định `formulary`), tắt rerank (`protocol = none`) và Langfuse.
- `dense` và `hybrid` dùng vector query có sẵn trong cache của `pharma-lab embed queries` (`--query-embeddings`, mặc định `data/cache/query_embeddings/<model-slug>.jsonl` theo `PHARMA_RETRIEVAL__EMBEDDING__MODEL`). Thiếu vector của query nào thì lệnh dừng trước khi retrieve; evaluation không bao giờ embed query qua endpoint. `bm25` không cần vector query.
- Phải có release đã import và publish; release lấy từ query đầu tiên và mọi hit sau phải cùng release.
- Run identity: evaluation path/sha256, collection Qdrant, `embedding_model`, `query_embeddings_sha256`, `retriever`, K, `limit`, `sample`, `sample_seed`, `release_id`, `chunker_version`.

| Retriever | Query embeddings | Qdrant | Candidate semantics |
| --- | --- | --- | --- |
| `bm25` | Không dùng | BM25 sparse | `candidate-k` là retrieval limit |
| `dense` | Bắt buộc, từ cache | dense | `candidate-k` là retrieval limit |
| `hybrid` | Bắt buộc, từ cache | dense + BM25 sparse, RRF | `prefetch-k` mỗi nhánh, `candidate-k` sau RRF |

```bash
uv run pharma-lab retrieve \
  --run backend-hybrid-qwen4b-p50-k30-rrf2 \
  --retriever hybrid \
  --prefetch-k 50 \
  --candidate-k 30 \
  --rrf-k 2
```

Candidate `chunk_id` là nhãn vị trí `<section_key>:chunk-<ordinal:03d>`, `document_text` là `embedding_text` của chunk (dùng cho rerank).

`--limit N` lấy N dòng đầu của file gold. `--sample N` lấy mẫu phân tầng theo `eval_group`: mỗi nhóm nhận số câu tỉ lệ với cỡ nhóm, làm tròn theo phần dư lớn nhất (phần dư bằng nhau thì nhóm xuất hiện trước trong file được ưu tiên), chọn ngẫu nhiên trong nhóm bằng `--sample-seed` (mặc định 0) và giữ thứ tự của file. Hai tuỳ chọn loại trừ nhau. Với bộ gold 10.000 câu, `--sample 1000` cho 500 `formulary`, 250 `leaflet`, 100 `chunk_risk`, 50 `patient_natural`, 50 `noisy_confuser`, 50 `multi_intent`.

```bash
uv run pharma-lab retrieve --run hybrid-qwen4b-p50-k30-rrf2-sample1000 --retriever hybrid --prefetch-k 50 --candidate-k 30 --rrf-k 2 --sample 1000 --sample-seed 0
```

## Rerank và metrics

`pharma-lab rerank` chỉ đọc candidates của run; với Kaggle, stage chỉ load reranker và chấm candidate pairs. Mỗi reranker có đúng một biến thể dưới `rerank/<model>/` của run. `--dry-run` in `target=` và `missing_pairs=N`, số cặp query–chunk chưa có điểm trong `data/cache/rerank_scores/<model>.jsonl`; khi cache đã đủ, rerank không khởi động model.

Với `--backend kaggle`, job chạy thành nhiều phiên GPU. `--kaggle-account auto` đọc `kaggle quota -v` của mọi profile trước mỗi phiên và chọn tài khoản còn nhiều giờ nhất (bằng nhau thì `accN` nhỏ hơn) mà không bị lệnh khác khoá; `--kaggle-account accN` giữ một tài khoản. Ngân sách phiên = min(`--budget-seconds`, quota còn lại − 0,5 giờ), tối thiểu 1 giờ nên `--budget-seconds` phải ≥ 3600. Lệnh dừng khi đủ cặp (`stop=complete`), hết quota (`stop=quota-exhausted`, exit 3, in bảng quota với `refresh_at`), một phiên không thêm cặp (`stop=no-progress`) hoặc đạt `--max-runs N` (`stop=max-runs`). Điểm của mọi phiên được gộp vào cache ở máy trước khi publish checkpoint, và cache đủ cặp thì lệnh ghi biến thể mà không gọi Kaggle. `--force` chấm lại từ đầu: xoá các cặp của run khỏi cache ở máy, phiên đầu bỏ qua checkpoint, kernel cũ và publish lại dataset, các phiên sau chỉ tiếp tục từ điểm của lần chạy này. Mỗi lệnh ghi log có timestamp vào `data/work/logs/rerank/<model>.log`.

`--recover-kernel OWNER/SLUG` (chỉ với `--backend kaggle`) tải output của một kernel đã kết thúc bằng profile có username `OWNER`, kiểm manifest (`rerank_scores`, đúng `--model`, đúng sha256) rồi gộp điểm vào `data/cache/rerank_scores/<model>.jsonl`, không nộp kernel mới. Điểm được khớp theo model, request contract, query và tài liệu, nên cách này dùng được cả khi job identity đã đổi. Khi cache đủ cặp của run, lệnh ghi luôn biến thể.

Rerank chỉ gọi `POST /v1/rerank` với file GGUF bản convert classifier; mỗi request là một câu hỏi cùng các ứng viên chưa có điểm, ở local lẫn Kaggle. `--backend local` dùng service `llama-reranker` của `../compose.yaml`: nếu `data/cache/local_profiles/rerank/<model>.json` khớp model, CPU (`/proc/cpuinfo`) và image llama.cpp của compose thì service được dựng theo profile, không thì theo mặc định của compose.

`--backend local --benchmark` đo từng mức của search space CPU: (`-np`, `-ub`) là (4, 4096), `-b` bằng `-ub`, `-c` bằng `-ub` nhân số slot cộng một, `--threads` 8/12. Mỗi mức dựng lại `llama-reranker`, chạy một nhóm khởi động không tính giờ rồi 6 nhóm câu hỏi × 15 ứng viên chọn phân tầng theo tổng số ký tự. Mức nào xếp hạng ứng viên khác mức hợp lệ đầu tiên thì bị loại (`rank_mismatch`); lệnh chọn p95 thấp nhất (bằng nhau thì số cặp/giây cao hơn), lưu profile và in `selected=`, `latency_p95_seconds=`, `env=`. Chép `env=` vào `.env` ở gốc repo và đặt `RERANK_TIMEOUT_SECONDS` ít nhất gấp đôi `latency_p95_seconds`.

```bash
uv run pharma-lab rerank --run hybrid-qwen4b-p50-k30-rrf2 --backend local --benchmark --model qwen3-reranker:4b-fp16
```

```bash
uv run pharma-lab rerank --run backend-hybrid-qwen4b-p50-k30-rrf2 --backend kaggle --model qwen3-reranker:4b-fp16 --kaggle-account auto
uv run pharma-lab metrics --run backend-hybrid-qwen4b-p50-k30-rrf2 --top-k 30
uv run pharma-lab metrics --run backend-hybrid-qwen4b-p50-k30-rrf2 --model qwen3-reranker:4b-fp16 --top-k 30
```

Report nằm ở `reports/baseline/top<K>-window<N>/` và `reports/rerank/<model>/top<K>-window<N>/`; chạy lại cùng cutoff với input khác thì lệnh dừng, `--force` thay report. Metrics gồm Hit@3/5/10/30, MRR, và với `answer_mode=multi_required`: Multi-section Recall@K, Multi-all-hit@K. `--top-k` không được lớn hơn `candidate-k`.

`pharma-lab metrics compare` so sánh hai reranker đã có report cùng cutoff trên cùng run. Lệnh ghép từng câu hỏi trong `metrics.jsonl` của hai report, bootstrap theo cặp (lấy mẫu lại có hoàn lại các câu hỏi `--resamples` lần bằng `--seed`) cho hiệu trung bình `candidate − baseline` của `--metric` (mặc định `mrr`, tức MRR@K), rồi in trung bình của từng model, hiệu, khoảng tin cậy 95% (`ci95_low`, `ci95_high`, phân vị 2,5% và 97,5%) và `decision`. `decision` là `--candidate` khi `ci95_low > 0`, ngược lại là `--baseline`. Lệnh dừng nếu hai report khác evaluation, candidates hoặc tập câu hỏi.

```bash
uv run pharma-lab metrics compare --run hybrid-qwen4b-p50-k30-rrf2-sample1000 --baseline qwen3-reranker:0.6b-fp16 --candidate qwen3-reranker:0.6b-fp16-vimed --metric mrr --top-k 30 --resamples 10000 --seed 0
```

`pharma-lab embed queries --backend local|kaggle --model MODEL` tạo cache vector query (`data/cache/query_embeddings/<model-slug>.jsonl`) mà `pharma-lab retrieve` dùng cho `dense` và `hybrid`; `--model` phải trùng model embedding của backend. Candidate `document_text` là `embedding_text` của hit, đúng văn bản reranker của benchmark cũ chấm.

## Runtime profiling trên Kaggle

Stage production tự benchmark một lần nếu chưa có profile hợp lệ và lưu ở `data/cache/kaggle_profiles/<workload>/<model>.json`; profile mất hiệu lực khi model, runtime, topology hoặc search space đổi. Mỗi mức là một cấu hình server đầy đủ (`-np`, `-ub`, số tài liệu mỗi request, số request đồng thời) và server khởi động lại khi đổi mức. Reranker chạy `--reranking --kv-unified -np N -c UB×(N+1) -b UB -ub UB`; tải đo là 32 nhóm câu hỏi × 30 ứng viên chọn phân tầng theo tổng số ký tự, cộng một nhóm khởi động không tính giờ. Mức nào xếp hạng ứng viên khác mức hợp lệ đầu tiên bị ghi `invalid` (`rank_mismatch`), mức chấm thiếu hoặc thừa cặp ghi `score_mismatch`; `max_abs_score_delta` vẫn được ghi cho mọi mức. Stage chọn mức có số cặp/giây cao nhất. Mọi mức đều lỗi thì stage dừng và in trạng thái cùng đuôi log server của từng mức.

## Archive dữ liệu

Mọi file dưới `data/` mà Git bỏ qua (trừ `data/work/`) được lưu trong dataset Kaggle private `<owner>/seed-pipeline-data`, `<owner>` là username của profile `--kaggle-account`. `push` đóng gói thành các phần `seed-pipeline-data.tar.zst.part-NNNN` (tối đa 1.900 MiB mỗi phần) cùng `archive-manifest.json` ghi sha256 từng phần và từng file, dựng ở `data/work/archive/`, rồi tạo dataset hoặc version mới.

```bash
uv run pharma-lab data push --kaggle-account acc1 --message "Rebuild evaluation runs"
uv run pharma-lab data pull --kaggle-account acc1
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

## Đánh giá end-to-end

Chi tiết trong [e2e-evaluation.md](e2e-evaluation.md).

| Command | Việc |
| --- | --- |
| `pharma-lab e2e golden sample [--seed S] [--force]` | Chọn 420 câu answerable (70 × 6 nhóm, phân tầng theo độ khó và answer mode) và 50 cặp multi-turn, ghi `data/evaluation/e2e/authoring/*.todo.jsonl` kèm text section và chunk trọng tâm |
| `pharma-lab e2e golden resample --slots IDS [--per-slot N]` | Đưa N câu chưa dùng cùng tầng cho các slot mà section gold không trả lời câu hỏi, ghi `replacement-NN.todo.jsonl` |
| `pharma-lab e2e golden check FILE` | Kiểm một lô `*.authored.jsonl` (quote nằm trong section, hành vi theo nhóm, `absent_terms` vắng trong corpus) |
| `pharma-lab e2e golden build` | Kiểm đủ số lượng từng nhóm rồi ghi `golden_e2e.jsonl` và manifest |
| `pharma-lab e2e run --run NAME --config full\|one-step\|no-judge-refine\|no-rephrase\|no-rerank [--limit N] [--concurrency N] [--deadline-seconds S] [--retry-errors]` | Chạy agent trên bộ golden, ghi `runs/<run>/<config>/answers.jsonl`; chạy lại để resume |
| `pharma-lab e2e judge --run NAME --config CONFIG [--judge-model MODEL] [--concurrency N] [--items IDS] [--force]` | Chấm bằng RAGAS và judge cấu trúc, ghi `judgments.jsonl` |
| `pharma-lab e2e calibration export --run NAME [--seed S]` | Xuất 100 câu mù để chấm hiệu chỉnh |
| `pharma-lab e2e calibration score --run NAME` | Tính κ và ρ giữa judge và `calibration/grades.jsonl` |
| `pharma-lab e2e report --run NAME` | Ghi bảng CSV/LaTeX và phân tích lỗi vào `runs/<run>/reports/` |
| `pharma-lab e2e rerank-server [--model MODEL] [--hours H] [--kaggle-account accN\|auto]` | Phục vụ reranker từ GPU Kaggle qua tunnel cloudflared có API key; ghi `data/work/serve/<model>.env` để các lượt `e2e run` dùng |
