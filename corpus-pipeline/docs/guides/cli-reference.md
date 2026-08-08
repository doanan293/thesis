# CLI reference

Public entrypoint:

```text
uv run corpus
```

Các command chính:

```text
corpus doctor
corpus build
corpus validate
corpus evaluation build
corpus embed chunks --backend local|kaggle
corpus embed queries --backend local|kaggle
corpus vectors upload
corpus retrieve --run NAME
corpus rerank --run NAME --backend local|kaggle
corpus metrics --run NAME
```

Global `--json` emits a machine-readable envelope and `--debug` enables
tracebacks. Dùng `uv run corpus COMMAND --help` để xem default thực tế.

## Retrieval modes

| Retriever | Query embeddings | Qdrant vectors | Candidate semantics |
| --- | --- | --- | --- |
| `bm25` | Không dùng | BM25 sparse | `candidate-k` là direct retrieval limit |
| `dense` | Bắt buộc, pre-embedded | Dense | `candidate-k` là direct retrieval limit |
| `hybrid` | Bắt buộc, pre-embedded | Dense + BM25 sparse | `prefetch-k` cho mỗi nhánh; `candidate-k` sau RRF |

BM25 không resolve query-embedding bundle. Dense và hybrid tự resolve bundle
theo evaluation file hash và embedding model, rồi validate completion, identity
và vector dimension trước khi query.

## Retrieval K parameters

```text
--prefetch-k INTEGER
--candidate-k INTEGER
--rrf-k INTEGER
```

- `--prefetch-k`: số kết quả lấy từ từng nhánh dense và BM25 trước hybrid fusion;
  chỉ hợp lệ với `--retriever hybrid`.
- `--candidate-k`: số candidates cuối được ghi vào retrieval artifact. Với dense
  và BM25 đây là retrieval limit; với hybrid đây là output limit sau RRF.
- `--rrf-k`: constant trong công thức RRF, mặc định `60`; không phải số
  candidates giữ lại.

Benchmark hybrid chuẩn:

```bash
uv run corpus retrieve \
  --run hybrid \
  --retriever hybrid \
  --prefetch-k 50 \
  --candidate-k 30 \
  --rrf-k 60
```

Nếu hybrid bỏ qua `--prefetch-k`, effective value bằng `candidate-k`. Value này
được lưu trong run identity và candidate manifest. Run cũ không có field này
được đọc theo cùng fallback để bảo đảm tương thích.

## Rerank và metrics

`corpus rerank` chỉ đọc complete candidate bundle của run. Với Kaggle backend,
stage chỉ load reranker model và score candidate pairs; không embed, retrieve,
truy cập Qdrant hoặc tính metrics.

`corpus metrics --run NAME --top-k 30` tạo baseline report. Nếu run có complete,
tương thích rerank score cache, command tạo thêm reranked report. Metrics gồm:

- Hit@3, Hit@5, Hit@10, Hit@30.
- MRR.
- Với `answer_mode=multi_required`: Multi-section Recall@3/5/10/30 và
  Multi-all-hit@3/5/10/30.

`--top-k` không được lớn hơn `candidate-k` của run identity.

## Artifacts và exit codes

Candidate và report artifacts được publish thành bundle có `manifest.json`.
Corpus embeddings, query embeddings và rerank scores là các cache JSONL phẳng:
`data/cache/vector_embeddings/<model-slug>.jsonl`,
`data/cache/query_embeddings/<model-slug>.jsonl` và
`data/cache/rerank_scores/<model-slug>.jsonl`. Mỗi record có checksum; cache
resume theo key và chỉ digest subset đang dùng được ghi vào run identity.
Retrieval, rerank và metrics dùng chung workspace named dưới
`data/runs/retrieval_eval/` qua `--run NAME`. Không trộn artifact candidate hoặc
input giữa các run/model/evaluation identity khác nhau.

| Code | Meaning |
| ---: | --- |
| 0 | Stage completed successfully |
| 1 | Runtime or validation failure |
| 2 | Invalid CLI usage |
| 3 | Resumable incomplete stage |
| 130 | Interrupted by the operator |
