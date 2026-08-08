# Local + Kaggle workflow

Workflow này giữ build, Qdrant, retrieval và metrics ở local; Kaggle chỉ chạy
các model-heavy stage. Với benchmark `hybrid + rerank`, local chuẩn bị hoàn tất
candidate bundle trước rồi Kaggle chỉ chấm reranker trên 30 candidates/query.

## 1. Kiểm tra môi trường

Khai báo Kaggle credentials/owners trong `.env`, sau đó chạy:

```bash
uv sync
uv run corpus doctor --backend kaggle
docker compose -f ../docker-compose.yml up -d qdrant
```

## 2. Build và evaluation ở local

```bash
uv run corpus build
uv run corpus validate
uv run corpus evaluation build
```

## 3. Embed corpus trên Kaggle, upload vector ở local

Dry-run để kiểm tra reconciliation:

```bash
uv run corpus embed chunks \
  --backend kaggle \
  --model qwen3-embedding:0.6b-fp16 \
  --dry-run
```

Chạy thật bằng cùng command không có `--dry-run`; chạy lại để resume nếu job bị
ngắt. Sau khi bundle corpus đã sync về local:

```bash
uv run corpus vectors upload --model qwen3-embedding:0.6b-fp16
```

## 4. Pre-embed query trên Kaggle

```bash
uv run corpus embed queries \
  --backend kaggle \
  --model qwen3-embedding:0.6b-fp16
```

Lệnh này tự động stream Kaggle kernel logs qua SSE trong cùng terminal, chờ
kernel hoàn tất, rồi tải artifact về local. Stream chỉ kết nối khi kernel bắt
đầu chạy và tự reconnect khi Kaggle trả lỗi tạm thời; status polling vẫn tiếp
tục độc lập. Không cần chạy riêng `kaggle kernels logs -f`.

Checkpoint được lưu theo model/runtime lineage. Khi input thay đổi, các query
hoặc candidate pair không đổi được tái sử dụng; record mới hoặc có hash thay
đổi mới chạy inference, còn record đã bị xóa sẽ không xuất hiện trong artifact
canonical. Journal là append-only nên resume không rewrite toàn bộ embedding
file. Với model topology replicated, worker gửi work đồng thời tới cả hai
server/GPU; topology sharded dùng một server thấy cả hai GPU.

Progress log có các nhóm `reusable`, `changed_or_new`, `deleted`, cùng
`recent_rate` và `average_rate`. Nếu kernel bị ngắt hoặc hết budget, lần chạy
lại đúng command sẽ tiếp tục từ các record đã commit trong checkpoint.

Sau khi query-embedding file đã merge vào
`data/cache/query_embeddings/<model-slug>.jsonl`, file này được dùng lại cho
dense và hybrid. BM25-only không cần query embeddings. Ba baseline retrieval
(`bm25`, `dense`, `hybrid`) đều chạy local; xem đầy đủ command trong
[local-only workflow](workflow-local-only.md).

## 5. Chuẩn bị hybrid candidate bundle ở local

Pipeline chuẩn cho rerank là:

```text
dense top 50 + BM25 top 50
  -> RRF (rrf-k=60), giữ top 30
  -> candidate bundle hoàn chỉnh ở local
  -> Kaggle reranker inference trên 30 candidates/query
  -> metrics baseline và reranked ở local
```

Chạy retrieval và baseline metrics:

```bash
uv run corpus retrieve \
  --run hybrid \
  --retriever hybrid \
  --prefetch-k 50 \
  --candidate-k 30 \
  --rrf-k 60

uv run corpus metrics --run hybrid --top-k 30
```

Candidate artifact phải complete trước khi submit rerank. Kaggle không embed
query, không truy cập Qdrant, không chạy lại RRF và không tính metrics.

## 6. Chỉ chạy reranker trên Kaggle

Kiểm tra dependency/checkpoint trước khi submit:

```bash
uv run corpus rerank \
  --run hybrid \
  --backend kaggle \
  --model qwen3-reranker:0.6b-fp16 \
  --dry-run
```

Chạy thật:

```bash
uv run corpus rerank \
  --run hybrid \
  --backend kaggle \
  --model qwen3-reranker:0.6b-fp16
```

Rerank stage chỉ nhận candidate JSONL cùng manifest và merge scores vào
`data/cache/rerank_scores/<model-slug>.jsonl`. Chạy lại cùng command để resume
các pair còn thiếu; cache được kiểm tra checksum trước khi merge.

## 7. Metrics offline ở local

Sau khi score file đã merge về local:

```bash
uv run corpus metrics --run hybrid --top-k 30
```

Report có baseline từ hybrid retrieval và reranked report từ cùng 30 candidates,
bao gồm Hit@3/5/10/30, MRR và multi-section metrics khi dataset có query
`multi_required`. Stage này không gọi Kaggle, model server hoặc Qdrant.

## Troubleshooting

- Credential/owner lỗi: sửa `.env`, rồi chạy lại `corpus doctor --backend kaggle`.
- Job pending hoặc hết budget: chạy lại đúng stage để reconciler resume.
- Candidate manifest không tương thích: dùng identity/run mới và không trộn
  artifact từ job khác. Score cache khác candidate set sẽ báo thiếu pair trước
  khi metrics chạy.
- `--top-k` lớn hơn 30: retrieve lại với candidate depth tương ứng.

Chi tiết command và semantics: [CLI reference](cli-reference.md). Chính sách
artifact downstream: [Downstream](downstream.md).
