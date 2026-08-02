# Local-only workflow

Workflow này dùng duy nhất public entrypoint `uv run corpus`. Mỗi stage xuất
artifact bundle có `manifest.json`; retrieval, rerank và metrics chia sẻ một
named run dưới `data/runs/retrieval_eval/`.

## 1. Kiểm tra môi trường

```bash
uv sync
uv run corpus doctor --backend local
```

Khởi động Qdrant trước các stage upload/retrieval:

```bash
docker compose -f ../docker-compose.yml up -d qdrant
```

## 2. Build và validate corpus

```bash
uv run corpus build
uv run corpus validate
```

Corpus được publish tại `data/processed/rag-final/`. Chỉ tiếp tục khi validation
thành công.

## 3. Build evaluation dataset

```bash
uv run corpus evaluation build
```

Output chuẩn là
`data/processed/evaluation/section_retrieval_eval.jsonl`.

## 4. Embed corpus và upload vectors

```bash
uv run corpus embed chunks \
  --backend local \
  --model qwen3-embedding:0.6b-fp16

uv run corpus vectors upload \
  --model qwen3-embedding:0.6b-fp16
```

`embed chunks` có thể resume bundle chưa hoàn chỉnh. `vectors upload` luôn từ
chối bundle thiếu records trước khi kết nối Qdrant.

## 5. Embed evaluation queries

```bash
uv run corpus embed queries \
  --backend local \
  --model qwen3-embedding:0.6b-fp16
```

Nếu bị ngắt, chạy lại cùng lệnh. Chỉ dùng `--force` khi chủ động thay thế kết
quả hiện có.

## 6. Smoke run

```bash
uv run corpus retrieve \
  --run local-smoke \
  --retriever hybrid \
  --candidate-k 50 \
  --limit 50

uv run corpus rerank \
  --run local-smoke \
  --backend local \
  --model qwen3-reranker:0.6b-fp16

uv run corpus metrics \
  --run local-smoke \
  --top-k 10
```

`run.json` đóng băng identity của run. Nếu thay model, retriever, limit hoặc
candidate count, dùng tên run mới; `--force` chỉ dùng khi thực sự muốn thay thế
identity đã lưu.

## 7. Full run

Sau khi smoke pass, dùng tên run riêng và bỏ `--limit`:

```bash
uv run corpus retrieve --run local-full --retriever hybrid --candidate-k 50
uv run corpus rerank --run local-full --backend local
uv run corpus metrics --run local-full --top-k 10
```

Metrics chỉ đọc frozen artifacts trong workspace; stage này không khởi động
model server hoặc truy cập Qdrant.

## Troubleshooting

- Xem flag và default thực tế bằng `uv run corpus COMMAND --help`.
- Nếu run identity xung đột, chọn `--run` mới thay vì tái sử dụng artifact cũ.
- Nếu embedding incomplete, chạy lại đúng stage với cùng model và output bundle.
- Nếu Qdrant thiếu collection, chạy lại `vectors upload` sau khi Qdrant healthy.

Chi tiết command và exit code: [CLI reference](cli-reference.md). Chính sách
artifact downstream: [Downstream](downstream.md).
