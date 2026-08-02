# Local + Kaggle workflow

Workflow này giữ build, Qdrant, retrieval và metrics ở local; các model-heavy
stage dùng cùng command với `--backend kaggle`. Không có Kaggle CLI public riêng.

## 1. Kiểm tra môi trường

Khai báo Kaggle credentials/owners trong `../.env`, sau đó chạy:

```bash
uv sync
uv run corpus doctor --backend kaggle
```

## 2. Build corpus và evaluation ở local

```bash
uv run corpus build
uv run corpus validate
uv run corpus evaluation build
```

## 3. Embed corpus trên Kaggle

Kiểm tra reconciliation mà không submit workload bằng shared flag `--dry-run`:

```bash
uv run corpus embed chunks \
  --backend kaggle \
  --model qwen3-embedding:0.6b-fp16 \
  --dry-run
```

Chạy thật:

```bash
uv run corpus embed chunks \
  --backend kaggle \
  --model qwen3-embedding:0.6b-fp16
```

CLI reconcile dependencies/checkpoints và publish bundle local chuẩn. Chạy lại
cùng command để resume.

## 4. Upload vectors vào Qdrant local

```bash
docker compose -f ../docker-compose.yml up -d qdrant
uv run corpus vectors upload --model qwen3-embedding:0.6b-fp16
```

Không cần copy đường dẫn JSONL thủ công; command resolve bundle chuẩn và kiểm tra
manifest/completion trước khi upload.

## 5. Embed queries trên Kaggle

```bash
uv run corpus embed queries \
  --backend kaggle \
  --model qwen3-embedding:0.6b-fp16
```

## 6. Retrieve ở local

```bash
uv run corpus retrieve \
  --run kaggle-smoke \
  --retriever hybrid \
  --candidate-k 50 \
  --limit 50
```

Candidate bundle được ghi vào workspace của run và tự động trở thành input cho
rerank.

## 7. Rerank trên Kaggle

```bash
uv run corpus rerank \
  --run kaggle-smoke \
  --backend kaggle \
  --model qwen3-reranker:0.6b-fp16 \
  --dry-run

uv run corpus rerank \
  --run kaggle-smoke \
  --backend kaggle \
  --model qwen3-reranker:0.6b-fp16
```

Rerank dùng candidates từ cùng run; không truyền lại candidate path trong
workflow thông thường.

## 8. Metrics offline ở local

```bash
uv run corpus metrics --run kaggle-smoke --top-k 10
```

Metrics xác thực candidate/score identity rồi sinh baseline và reranked reports,
không gọi Kaggle, Qdrant hoặc model server.

## 9. Full run

Sau khi smoke pass, dùng tên run khác và bỏ `--limit`:

```bash
uv run corpus retrieve --run kaggle-full --retriever hybrid --candidate-k 50
uv run corpus rerank --run kaggle-full --backend kaggle
uv run corpus metrics --run kaggle-full --top-k 10
```

## Troubleshooting

- Credential/owner lỗi: sửa `../.env`, rồi chạy lại `corpus doctor --backend kaggle`.
- Job pending hoặc hết budget: chạy lại cùng stage; compatible checkpoint được reuse.
- Artifact identity mismatch: dùng tên run mới hoặc rebuild stage upstream.
- Xem flag chuẩn bằng `uv run corpus COMMAND --help`.

Chi tiết command và exit code: [CLI reference](cli-reference.md). Chính sách
artifact downstream: [Downstream](downstream.md).
