# Local-only workflow

Workflow này dùng duy nhất public entrypoint `uv run seed`. Qdrant vẫn chạy
local, nhưng sau khi seed vectors và query embeddings đã hoàn chỉnh thì các
stage retrieval và metrics không cần model server, Kaggle hoặc Internet.

Retrieval, rerank và metrics chia sẻ một named run dưới
`data/retrieval_eval/`, với payload lớn tương ứng dưới `data/heavy/retrieval_eval/`.
Mỗi run đóng băng evaluation input, model,
retriever, K và query-embedding identity.

## 1. Kiểm tra môi trường

```bash
uv sync
uv run seed doctor --backend local
docker compose -f ../docker-compose.yml up -d qdrant
```

## 2. Chuẩn bị artifact một lần

```bash
uv run seed build
uv run seed validate
uv run seed evaluation build

uv run seed embed chunks \
  --backend local \
  --model qwen3-embedding:4b-fp16

uv run seed vectors upload \
  --model qwen3-embedding:4b-fp16

uv run seed embed queries \
  --backend local \
  --model qwen3-embedding:4b-fp16
```

`embed queries` ghi vào một file duy nhất theo model tại
`data/heavy/cache/query_embeddings/qwen3_embedding_0_6b_fp16.jsonl`. File này được
dùng lại cho `dense` và `hybrid`; BM25-only không cần query cache. Nếu stage bị
ngắt, chạy lại cùng command để resume. Mỗi dòng có checksum và được kiểm tra
trước khi retrieval sử dụng.

## 3. Đánh giá ba retrieval baseline

Mỗi baseline cần ít nhất 30 candidates/query để tính Hit@3, Hit@5, Hit@10 và
Hit@30. Metrics cũng sinh MRR; các query `multi_required` có thêm
Multi-section Recall và Multi-all-hit tại cùng các cutoff.

### BM25-only

```bash
uv run seed retrieve \
  --run bm25-qwen4b-k30 \
  --retriever bm25 \
  --model qwen3-embedding:4b-fp16 \
  --candidate-k 30

uv run seed metrics \
  --run bm25-qwen4b-k30 \
  --top-k 30
```

BM25 không dùng query embedding. `--model` ở đây vẫn chọn collection
`thesis_chunks_qwen3_embedding_0_6b_fp16`, nơi mỗi point có cả BM25 sparse
vector.

### Dense-only

```bash
uv run seed retrieve \
  --run dense-qwen4b-k30 \
  --retriever dense \
  --model qwen3-embedding:4b-fp16 \
  --candidate-k 30

uv run seed metrics \
  --run dense-qwen4b-k30 \
  --top-k 30
```

Dense validate và dùng lại query-embedding cache tương ứng với cùng embedding
model `qwen3-embedding:4b-fp16`.

### Hybrid

Hybrid lấy dense top 50 và BM25 top 50, sau đó Qdrant dùng RRF với constant 2
và chỉ giữ top 30 trong candidate artifact:

```bash
uv run seed retrieve \
  --run hybrid-qwen4b-p50-k30-rrf2 \
  --retriever hybrid \
  --model qwen3-embedding:4b-fp16 \
  --prefetch-k 50 \
  --candidate-k 30 \
  --rrf-k 2

uv run seed metrics \
  --run hybrid-qwen4b-p50-k30-rrf2 \
  --top-k 30
```

`--prefetch-k` là số kết quả của từng nhánh; `--candidate-k` là số kết quả sau
fusion. Không truyền `--prefetch-k` sẽ dùng cùng giá trị với `--candidate-k` để
giữ tương thích với run cũ.

## 4. Hybrid + rerank

Không chạy hybrid retrieval lần thứ hai. Run `hybrid` đã có candidate bundle
đúng 30 candidates/query. Có thể rerank local như sau:

```bash
uv run seed rerank \
  --run hybrid-qwen4b-p50-k30-rrf2 \
  --backend local \
  --model qwen3-reranker:0.6b-fp16
uv run seed metrics \
  --run hybrid-qwen4b-p50-k30-rrf2 \
  --top-k 30
```

Candidate bundle không bị thay đổi khi đổi reranker. Mỗi model/revision tạo
score variant riêng dưới run. Metrics mặc định tính baseline và mọi variant đã
hoàn tất; dùng `--model` hoặc `--variant` để lọc. Không cần tự chọn output
directory:

```bash
uv run seed rerank \
  --run hybrid-qwen4b-p50-k30-rrf2 \
  --backend local \
  --model bge-reranker-v2-m3:f16

uv run seed metrics \
  --run hybrid-qwen4b-p50-k30-rrf2 \
  --model bge-reranker-v2-m3:f16 \
  --top-k 30
```

Global score cache chỉ phục vụ resume; artifact chuẩn nằm trong run và không bị
ghi đè khi chạy model khác.

## 5. Smoke test và resume

Thêm `--limit 50` vào từng lệnh `retrieve` để kiểm tra nhanh trước khi bỏ giới
hạn. Ví dụ, dùng `hybrid-qwen4b-p50-k30-rrf2-smoke50` cho smoke run và
`hybrid-qwen4b-p50-k30-rrf2` cho benchmark đầy đủ; bỏ `--limit` làm thay đổi
run identity. Không trộn candidates, query bundle hoặc model giữa các run; nếu
identity thay đổi, chọn tên run mới hoặc dùng `--force` có chủ đích.

Metrics chỉ đọc frozen artifacts trong workspace và không khởi động model server
hoặc truy cập Qdrant.

## Troubleshooting

- Qdrant thiếu collection: kiểm tra service healthy rồi chạy lại `vectors upload`.
- Dense/hybrid thiếu query cache: chạy hoàn chỉnh `seed embed queries` với
  đúng evaluation file và model.
- BM25 không cần query bundle; chỉ cần collection có sparse vector BM25.
- `--top-k` lớn hơn `candidate-k`: retrieve lại với candidate depth đủ lớn.
- Run identity xung đột: chọn `--run` mới thay vì ghi đè artifact cũ.

Chi tiết flags và metric semantics: [CLI reference](cli-reference.md). Chính sách
artifact downstream: [Downstream](downstream.md).
