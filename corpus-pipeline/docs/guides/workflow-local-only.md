# Local-only workflow

Workflow này dùng duy nhất public entrypoint `uv run corpus`. Qdrant vẫn chạy
local, nhưng sau khi corpus vectors và query embeddings đã hoàn chỉnh thì các
stage retrieval và metrics không cần model server, Kaggle hoặc Internet.

Retrieval, rerank và metrics chia sẻ một named run dưới
`data/runs/retrieval_eval/`. Mỗi run đóng băng evaluation input, model,
retriever, K và query-embedding identity.

## 1. Kiểm tra môi trường

```bash
uv sync
uv run corpus doctor --backend local
docker compose -f ../docker-compose.yml up -d qdrant
```

## 2. Chuẩn bị artifact một lần

```bash
uv run corpus build
uv run corpus validate
uv run corpus evaluation build

uv run corpus embed chunks \
  --backend local \
  --model qwen3-embedding:0.6b-fp16

uv run corpus vectors upload \
  --model qwen3-embedding:0.6b-fp16

uv run corpus embed queries \
  --backend local \
  --model qwen3-embedding:0.6b-fp16
```

`embed queries` ghi vào một file duy nhất theo model tại
`data/cache/query_embeddings/qwen3_embedding_0_6b_fp16.jsonl`. File này được
dùng lại cho `dense` và `hybrid`; BM25-only không cần query cache. Nếu stage bị
ngắt, chạy lại cùng command để resume. Mỗi dòng có checksum và được kiểm tra
trước khi retrieval sử dụng.

## 3. Đánh giá ba retrieval baseline

Mỗi baseline cần ít nhất 30 candidates/query để tính Hit@3, Hit@5, Hit@10 và
Hit@30. Metrics cũng sinh MRR; các query `multi_required` có thêm
Multi-section Recall và Multi-all-hit tại cùng các cutoff.

### BM25-only

```bash
uv run corpus retrieve \
  --run bm25 \
  --retriever bm25 \
  --candidate-k 30

uv run corpus metrics --run bm25 --top-k 30
```

### Dense-only

```bash
uv run corpus retrieve \
  --run dense \
  --retriever dense \
  --candidate-k 30

uv run corpus metrics --run dense --top-k 30
```

### Hybrid

Hybrid lấy dense top 50 và BM25 top 50, sau đó Qdrant dùng RRF với constant 60
và chỉ giữ top 30 trong candidate artifact:

```bash
uv run corpus retrieve \
  --run hybrid \
  --retriever hybrid \
  --prefetch-k 50 \
  --candidate-k 30 \
  --rrf-k 60

uv run corpus metrics --run hybrid --top-k 30
```

`--prefetch-k` là số kết quả của từng nhánh; `--candidate-k` là số kết quả sau
fusion. Không truyền `--prefetch-k` sẽ dùng cùng giá trị với `--candidate-k` để
giữ tương thích với run cũ.

## 4. Hybrid + rerank

Không chạy hybrid retrieval lần thứ hai. Run `hybrid` đã có candidate bundle
đúng 30 candidates/query. Có thể rerank local như sau:

```bash
uv run corpus rerank --run hybrid --backend local
uv run corpus metrics --run hybrid --top-k 30
```

Metrics lần đầu là `baseline.md`; sau khi có rerank scores trong
`data/cache/rerank_scores/<model-slug>.jsonl`, lần chạy lại tạo thêm
`reranked.md` từ cùng candidate artifact.

## 5. Smoke test và resume

Thêm `--limit 50` vào từng lệnh `retrieve` để kiểm tra nhanh trước khi bỏ giới
hạn. Dùng tên run riêng cho mỗi cấu hình. Không trộn candidates, query bundle
hoặc model giữa các run; nếu identity thay đổi, chọn tên run mới hoặc dùng
`--force` có chủ đích.

Metrics chỉ đọc frozen artifacts trong workspace và không khởi động model server
hoặc truy cập Qdrant.

## Troubleshooting

- Qdrant thiếu collection: kiểm tra service healthy rồi chạy lại `vectors upload`.
- Dense/hybrid thiếu query cache: chạy hoàn chỉnh `corpus embed queries` với
  đúng evaluation file và model.
- BM25 không cần query bundle; chỉ cần collection có sparse vector BM25.
- `--top-k` lớn hơn `candidate-k`: retrieve lại với candidate depth đủ lớn.
- Run identity xung đột: chọn `--run` mới thay vì ghi đè artifact cũ.

Chi tiết flags và metric semantics: [CLI reference](cli-reference.md). Chính sách
artifact downstream: [Downstream](downstream.md).
