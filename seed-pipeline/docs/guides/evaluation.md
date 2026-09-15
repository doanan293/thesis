# Đánh giá retrieval

Guide này dựng lại các số liệu retrieval của luận văn từ dữ liệu trong archive. Lệnh `uv run seed` chạy trong `seed-pipeline/`, lệnh `uv run pharma-agent` chạy trong `backend/`, lệnh `docker compose` chạy ở thư mục gốc repo.

## 1. Dữ liệu

```bash
uv run seed data pull --kaggle-account acc1
```

Lệnh tải bundle `data/corpus/formulary` (embedding của 5 model), bộ gold `data/evaluation/gold`, candidates và artifact rerank của các run, cache vector query và điểm rerank, rồi kiểm sha256 từng file. `run.json`, manifest và `report.md` của các run đã có trong Git.

## 2. Môi trường riêng cho đánh giá

```bash
POSTGRES_PORT=5434 QDRANT_HTTP_PORT=6335 QDRANT_GRPC_PORT=6336 docker compose -p thesis-eval up -d --wait postgres qdrant
export PHARMA_POSTGRES__DSN=postgresql+psycopg://thesis:thesis@localhost:5434/thesis
export PHARMA_QDRANT__URL=http://localhost:6335
```

Stack `thesis-eval` tách khỏi stack của app (cổng 5433 và 6333). Không cần llama.cpp: vector chunk có sẵn trong bundle, vector query có sẵn trong cache, `seed retrieve` tắt rerank. Xong việc: `docker compose -p thesis-eval down -v`.

## 3. Chạy một model

Mỗi model đọc và ghi alias Qdrant riêng `eval_<model_slug>` qua `PHARMA_RETRIEVAL__QDRANT_COLLECTION`. `pharma-agent corpus import` tạo alias khi chưa có nhưng không chuyển một alias đã có sang collection của model khác, nên dùng chung một alias cho nhiều model sẽ truy vấn nhầm vector. Import và publish model trước khi retrieve, vì `seed retrieve` đọc release đang publish.

```bash
export PHARMA_RETRIEVAL__EMBEDDING__MODEL=qwen3-embedding:4b-fp16
export PHARMA_RETRIEVAL__EMBEDDING__DIMENSION=2560
export PHARMA_RETRIEVAL__QDRANT_COLLECTION=eval_qwen3_embedding_4b_fp16
```

Trong `backend/`:

```bash
uv run pharma-agent corpus import ../seed-pipeline/data/corpus/formulary --collection formulary --publish
```

Trong `seed-pipeline/`:

```bash
uv run seed retrieve --run dense-qwen4b-k30 --retriever dense --candidate-k 30 --force
uv run seed retrieve --run bm25-qwen4b-k30 --retriever bm25 --candidate-k 30 --force
uv run seed retrieve --run hybrid-qwen4b-p50-k30-rrf60 --retriever hybrid --prefetch-k 50 --candidate-k 30 --rrf-k 60 --force
uv run seed metrics --run hybrid-qwen4b-p50-k30-rrf60 --top-k 30 --force
```

`--force` thay run tải từ archive, vì stack mới có release id khác. Mỗi lần retrieve 10.000 query mất khoảng 10–20 phút.

| Model | Chiều vector | `PHARMA_RETRIEVAL__QDRANT_COLLECTION` | Run |
| --- | ---: | --- | --- |
| `embeddinggemma:300m` | 768 | `eval_embeddinggemma_300m` | `dense-gemma300m-k30` |
| `bge-m3:567m-fp16` | 1024 | `eval_bge_m3_567m_fp16` | `dense-bge-m3-k30` |
| `qwen3-embedding:0.6b-fp16` | 1024 | `eval_qwen3_embedding_0_6b_fp16` | `dense-qwen06b-k30` |
| `qwen3-embedding:8b-fp16` | 4096 | `eval_qwen3_embedding_8b_fp16` | `dense-qwen8b-k30` |
| `qwen3-embedding:4b-fp16` | 2560 | `eval_qwen3_embedding_4b_fp16` | `dense-qwen4b-k30`, `bm25-qwen4b-k30`, `hybrid-qwen4b-p50-k30-rrf2`, `hybrid-qwen4b-p50-k30-rrf60` |

## 4. Rerank

```bash
uv run seed rerank --run hybrid-qwen4b-p50-k30-rrf2 --model qwen3-reranker:4b-fp16 --dry-run
uv run seed rerank --run hybrid-qwen4b-p50-k30-rrf2 --model qwen3-reranker:4b-fp16
uv run seed metrics --run hybrid-qwen4b-p50-k30-rrf2 --model qwen3-reranker:4b-fp16 --top-k 30
```

`--dry-run` in `missing_pairs=N`. Candidates của `hybrid-qwen4b-p50-k30-rrf2` trong archive là đúng các cặp query–chunk đã có điểm trong `data/cache/rerank_scores/`, nên `N` bằng 0 và rerank chỉ đọc cache, không khởi động model.

Mọi reranker được gọi qua `/v1/rerank`. Kết quả `bge-reranker-v2-gemma:f16` ở mục 6 là kết quả cuối cùng: model đã bỏ khỏi catalog, report của nó vẫn nằm ở `reports/rerank/bge_reranker_v2_gemma_f16/`.

Cấu hình `llama-reranker` cho backend trên CPU được chọn bằng benchmark đầu-cuối trên máy production (chạy trong tmux: mỗi mức nạp lại model 4B trên CPU):

```bash
uv run seed rerank --run hybrid-qwen4b-p50-k30-rrf2 --backend local --benchmark --model qwen3-reranker:4b-fp16
```

Lệnh lưu `data/cache/local_profiles/rerank/<model>.json` và in các biến `LLAMA_RERANKER_*` cho `.env` ở gốc repo. Trên Kaggle, benchmark chạy tự động trước lần chấm đầu tiên khi chưa có profile khớp (xem [CLI reference](cli-reference.md)).

Qdrant tìm dense bằng HNSW (gần đúng), nên retrieve lại trên index dựng mới có thể chọn khác vài chunk có điểm sát nhau ở cuối top 30. Lần dựng lại tháng 9/2026 lệch 4.678 trên 300.000 cặp, gần hết ở hạng 21–30. Khi đó `N` lớn hơn 0 và cần chạy reranker (CPU hoặc Kaggle) cho các cặp còn thiếu.

## 5. Run `dense-text-embedding-3-large-k30`

Run này dùng embedding API trả phí nên không chạy lại. Candidates được nhập từ lần chạy gốc (`origin: imported` trong `run.json`) và chỉ tính lại metrics.

## 6. Kết quả tham chiếu

| Run / biến thể | Hit@10 | MRR |
| --- | ---: | ---: |
| `bm25-qwen4b-k30` | 77,97% | 0,5550 |
| `dense-gemma300m-k30` | 83,80% | 0,5902 |
| `dense-bge-m3-k30` | 90,95% | 0,7334 |
| `dense-qwen06b-k30` | 91,88% | 0,7042 |
| `dense-qwen4b-k30` | 95,22% | 0,7786 |
| `dense-qwen8b-k30` | 95,26% | 0,7952 |
| `dense-text-embedding-3-large-k30` | 94,85% | 0,7506 |
| `hybrid-qwen4b-p50-k30-rrf60` | 92,44% | 0,6786 |
| `hybrid-qwen4b-p50-k30-rrf2` | 95,67% | 0,7242 |
| rrf2 + `bge-reranker-v2-gemma:f16` | 95,30% | 0,7570 |
| rrf2 + `bge-reranker-v2-m3:f16` | 95,04% | 0,7982 |
| rrf2 + `qwen3-reranker:0.6b-fp16` | 96,18% | 0,7823 |
| rrf2 + `qwen3-reranker:4b-fp16` | 96,98% | 0,8060 |
| rrf2 + `qwen3-reranker:8b-fp16` | 89,75% | 0,4797 |

Số trước khi chuyển layout và bảng so sánh nằm trong commit "rebuild the evaluation runs with the new CLI". Lần dựng lại chấp nhận Hit@10 thấp hơn tối đa 1 điểm phần trăm và MRR thấp hơn tối đa 0,01.
