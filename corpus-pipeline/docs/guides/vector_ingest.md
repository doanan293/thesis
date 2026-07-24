# Hướng Dẫn Ingestion Vector & Cache Qdrant

Tài liệu này chi tiết kiến trúc lưu trữ vector Qdrant, quản lý Postgres app state, quy trình build metadata, và các phương thức nạp vector (Local llama.cpp Docker & Kaggle 2×T4 CUDA runs).

---

## 1. Kiến Trúc Qdrant-Only Corpus + Postgres App

Pipeline dùng artifact lean Qdrant-ready cho ingest:

- `data/processed/rag-final/sections.jsonl`: Giữ full section text offline để kiểm toán, rebuild, và đối chiếu.
- `data/processed/rag-final/chunks.jsonl`: Giữ chunk source sau canonical build, gồm các field build/hydrate chi tiết.
- `data/processed/rag-final/chunks.jsonl`: Artifact runtime ở mức chunk, đọc trực tiếp bởi Qdrant ingest.

Qdrant ingest tạo point trực tiếp từ `chunks.jsonl`. Final artifacts không lưu wrapper ingest riêng.

Qdrant dùng lean runtime payload ở mức chunk: `chunk_id`, `section_id`, `chunk_index`, `hydrate_strategy`, `source`, `title`, `section`, `start_page`, `end_page`, `context_header`, `payload.chunk_text`, `payload.embedding_text`, compact `colloquial_mapping`, compact `term_annotations`, và optional `content_type`/`table_id`.
Trường `source` chứa nhãn citation: Dược thư dùng `Dược thư Quốc gia Việt Nam (Xuất bản lần thứ 2) – Nhà xuất bản Y học, Hà Nội, 2018`; tờ hướng dẫn dùng `Tờ hướng dẫn sử dụng`.

Postgres app schema nằm tại `src/postgres_store/schema/rag_app_schema.sql`. Schema này chỉ dành cho app/chat/runtime state như user, session, message, retrieval run, retrieval hit, và feedback. Corpus text không được import vào Postgres production.

---

## 2. Chuẩn Bị Metadata & Ingest

### 2.1 Build Metadata Artifacts

```bash
uv run python -m cli.build_rag_metadata \
  --corpus-version rag-final-2026-07-05
```

Lệnh này kết xuất:
- `data/processed/rag-final/sections.jsonl`: Full section text offline và hydrate metadata.
- `data/processed/rag-final/chunks.jsonl`: Chunk source offline.
- `data/processed/rag-final/chunks.jsonl`: Lean Qdrant-ready runtime rows.
- `data/processed/rag-final/term_enrichment_audit.json`: Báo cáo coverage glossary và thuật ngữ viết tắt chưa map.

---

### 2.2 Sinh Vector Và Nạp Vào Qdrant

Model runtime thống nhất là `llama.cpp`. Tất cả file GGUF đặt tại:

```text
/home/andv/personal/thesis/ai-models/gguf
```

Catalog chuẩn nằm tại `src/model_runtime/catalog.py`. Runtime không tự tải, convert, quantize hay đổi tên model.

#### 2.2.1 Local CPU bằng Docker Compose

`../docker-compose.yml` khai báo 2 service độc lập dùng image `ghcr.io/ggml-org/llama.cpp:server`:

| Service | Host port | Vai trò |
|---|---:|---|
| `llama-embedding` | 11434 | `/v1/embeddings` |
| `llama-reranker` | 11435 | native rerank hoặc completion probabilities |

Cả hai service mặc định CPU-only. Kiểm tra Compose:

```bash
docker compose -f ../docker-compose.yml config --quiet
```

#### 2.2.2 Chạy All-In-One Trên Máy Local

Lệnh mặc định đọc `chunks.jsonl`, sinh hoặc tái sử dụng cache, rebuild collection tương ứng và upload dense + BM25 sparse vectors:

```bash
uv run python -m cli.ingest_vectors \
  --model qwen3-embedding:0.6b-fp16
```

Các model embedding hợp lệ:
- `embeddinggemma:300m`
- `bge-m3:567m-fp16`
- `qwen3-embedding:0.6b-fp16`
- `qwen3-embedding:4b-fp16`
- `qwen3-embedding:8b-fp16`

Collection tên theo model, ví dụ: `thesis_chunks_qwen3_embedding_0_6b_fp16`.

Tham số thường dùng:
```bash
uv run python -m cli.ingest_vectors \
  --model embeddinggemma:300m \
  --input-batch-size 1 \
  --qdrant-batch-size 1000 \
  --request-timeout 900
```

#### 2.2.3 Chỉ Sinh Cache Embedding Local

```bash
uv run python -m cli.ingest_vectors \
  --model qwen3-embedding:0.6b-fp16 \
  --cache-only \
  --max-runtime-seconds 3600 \
  --stop-margin-seconds 120
```

Cache mặc định lưu tại `data/cache/vector_embeddings/<model_slug>.jsonl`.

#### 2.2.4 Dùng llama.cpp Server Bên Ngoài

```bash
uv run python -m cli.ingest_vectors \
  --model embeddinggemma:300m \
  --server-mode external \
  --llama-server-url http://127.0.0.1:8080
```

#### 2.2.5 Import Qdrant Từ Cache Không Cần Khởi Động Model

Khi cache đã hoàn chỉnh toàn bộ corpus:

```bash
uv run python -m cli.ingest_vectors \
  --model qwen3-embedding:0.6b-fp16 \
  --upload-from-cache
```

---

### 2.3 Kaggle 2×T4: Cloud-First Vector Cache

Kaggle CLI tự load `KAGGLE_USERNAME` từ `/home/andv/personal/thesis/.env`.
Corpus, model GGUF, runtime, checkpoint và trạng thái hoàn thành đều nằm trên
Kaggle. Các lệnh cloud dùng workspace ngẫu nhiên dưới `/tmp` và tự dọn; không
tạo lại `.kaggle/` trong repository.

#### Publish dependency tĩnh bằng lệnh riêng:

```bash
uv run python -m cli.kaggle_vector_cache corpus publish
uv run python -m cli.kaggle_vector_cache models publish \
  --gguf-root /home/andv/personal/thesis/ai-models/gguf
uv run python -m cli.kaggle_vector_cache llama-cpp-runtime build
```

`run` không tự upload lại corpus/model/runtime. Nếu dependency chưa có trên
Kaggle, lệnh sẽ dừng và in lệnh publish cần chạy.

#### Chạy hoặc resume kernel trên Kaggle:

```bash
uv run python -m cli.kaggle_vector_cache run \
  --model qwen3-embedding:8b-fp16 \
  --repeat-until-complete \
  --total-budget-seconds 10800 \
  --export-reserve-seconds 900
```

Checkpoint trung gian được chuyển qua `/tmp`, kiểm tra checksum, upload thành
dataset version rồi xóa. Lần chạy sau resume từ checkpoint cloud; không cần
cache hoặc manifest local.

#### Theo dõi trạng thái cloud và benchmark:

```bash
uv run python -m cli.kaggle_vector_cache status \
  --model embeddinggemma:300m

uv run python -m cli.kaggle_vector_cache benchmark \
  --model bge-m3:567m-fp16 \
  --total-budget-seconds 600
```

#### Chỉ download khi thực sự cần cache local:

```bash
uv run python -m cli.kaggle_vector_cache download \
  --checkpoint-owner USER_CU \
  --model embeddinggemma:300m \
  --output-dir data/cache/vector_embeddings
```

`download` là lệnh duy nhất tạo cache persistent trong project. Download được
thực hiện vào `/tmp`, validate trước, rồi atomic-promote cache và manifest.

#### Log Streaming & Resume Policy:

CLI local vẫn poll log snapshot và stream progress. Các file tạm nằm trong
workspace `/tmp/corpus-pipeline-kaggle-<random>/`:
```text
[00:00:00] RUNNING
[00:01:00] RUNNING | elapsed=1m00s | waiting for new Kaggle worker logs
[00:01:18] Worker gpu_ready | devices=2xT4
[00:01:42] Worker server_ready | topology=replicated_2x1 | parallel=4 | batch=32 | servers=2
[00:02:05] Embedding progress: 4096/20006 (20.5%), speed=145.0 chunks/s, ETA=1m50s
[00:04:11] Worker checkpoint | complete=20006 | total=20006 | missing=0
[00:04:15] COMPLETE
```
Nếu run hết budget, checkpoint được publish lên Kaggle; lượt run sau tự động
mount checkpoint cloud để tiếp tục. Các thư mục `.kaggle/` cũ từ phiên bản
trước không được tự động xóa; chỉ xóa sau khi đã xác nhận dữ liệu cần thiết đã
ở trên Kaggle.
