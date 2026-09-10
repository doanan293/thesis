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

## Run, retriever và model

`--run` dùng cùng một tên dưới hai cây: metadata ở
`data/retrieval_eval/<run-name>/` và payload lớn ở
`data/heavy/retrieval_eval/<run-name>/`. Retrieval, rerank và metrics dùng tên
này để mở lại cùng một workspace và các artifact đã đóng băng; tên run không tự
chọn thuật toán.

`--retriever` độc lập chọn cách tạo candidates: `bm25`, `dense` hoặc `hybrid`.
`--model` của `corpus retrieve` chọn embedding identity và Qdrant collection
tương ứng. BM25 không embed query nhưng vẫn resolve collection đó, vì các điểm
trong collection có BM25 sparse vector. `--model` của `corpus rerank` là
reranker model, không phải embedding model. `corpus metrics` chỉ đọc artifact
đã có và không chạy model.

| Run name | Retriever | Important parameters |
| --- | --- | --- |
| `bm25-qwen06b-k30` | `bm25` | embedding collection `qwen06b`, `candidate-k=30` |
| `dense-qwen06b-k30` | `dense` | embedding `qwen06b`, `candidate-k=30` |
| `hybrid-qwen06b-p50-k30-rrf2` | `hybrid` | embedding `qwen06b`, `prefetch-k=50`, `candidate-k=30`, `rrf-k=2` |

Run identity còn đóng băng evaluation path/checksum, collection, query-embedding
digest, limit và các K parameters. Khi một identity field thay đổi, nên dùng
tên run mới thay vì dùng `--force` một cách tùy ý.

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
- `--rrf-k`: constant trong công thức RRF, mặc định `2`; không phải số
  candidates giữ lại.

Benchmark hybrid chuẩn:

```bash
uv run corpus retrieve \
  --run hybrid-qwen06b-p50-k30-rrf2 \
  --retriever hybrid \
  --model qwen3-embedding:0.6b-fp16 \
  --prefetch-k 50 \
  --candidate-k 30 \
  --rrf-k 2
```

Nếu hybrid bỏ qua `--prefetch-k`, effective value bằng `candidate-k`. Value này
được lưu trong run identity và candidate manifest. Run cũ không có field này
được đọc theo cùng fallback để bảo đảm tương thích.

## Rerank và metrics

`corpus rerank` chỉ đọc complete candidate bundle của run. Với Kaggle backend,
stage chỉ load reranker model và score candidate pairs; không embed, retrieve,
truy cập Qdrant hoặc tính metrics. Rerank gắn score vào run hiện có, không chạy
lại retrieval và không tạo một retrieval run thứ hai.

Ở profile mode (`--kaggle-account accN`), rerank và query/corpus production
stages tự tìm checkpoint tương thích trong các profile `accN`, chọn completion
lớn nhất và chỉ mirror khi source có tiến bộ strict hơn target. Mirror dùng
credential riêng của target, source dataset giữ nguyên; target phải `READY` và
được validate lại trước khi kernel submit. `--dry-run` chỉ báo kế hoạch, còn
`--force` không thực hiện inheritance.

```bash
uv run corpus rerank \
  --run hybrid-qwen06b-p50-k30-rrf2 \
  --backend local \
  --model qwen3-reranker:0.6b-fp16

uv run corpus metrics \
  --run hybrid-qwen06b-p50-k30-rrf2 \
  --top-k 30
```

Một run dùng chung candidate bundle bất biến cho mọi reranker. Mỗi lần rerank
hoàn tất tạo một score variant bất biến dưới run; đổi model, GGUF SHA hoặc prompt
contract sẽ tạo variant mới. Không cần truyền `--output-dir`.

Metrics mặc định tạo/reuse baseline và report cho tất cả reranker variants đã
hoàn tất. Có thể lọc theo model hoặc đúng variant:

```bash
uv run corpus metrics --run hybrid-qwen06b-p50-k30-rrf2 --top-k 30
uv run corpus metrics --run hybrid-qwen06b-p50-k30-rrf2 \
  --model qwen3-reranker:0.6b-fp16 --top-k 30
uv run corpus metrics --run hybrid-qwen06b-p50-k30-rrf2 \
  --variant VARIANT_SHA256_PREFIX --top-k 30
```

`--model` và `--variant` loại trừ nhau. `--output-dir` của rerank/metrics còn
được chấp nhận tạm thời cho script cũ nhưng chỉ phát cảnh báo; output chuẩn
luôn do run tự sinh. Metrics gồm:

- Hit@3, Hit@5, Hit@10, Hit@30.
- MRR.
- Với `answer_mode=multi_required`: Multi-section Recall@3/5/10/30 và
  Multi-all-hit@3/5/10/30.

`--top-k` không được lớn hơn `candidate-k` của run identity.

### Runtime profiling

Khi chạy Kaggle production, hệ thống tự benchmark workload tương ứng đúng một
lần nếu chưa có profile hợp lệ. Profile được lưu dưới `data/heavy/runtime_kaggle_profiles/`
và các lần sau sẽ tái sử dụng; profile bị vô hiệu khi model, runtime, topology
hoặc search space thay đổi. Benchmark luôn dùng 512 mẫu và lỗi benchmark sẽ
dừng pipeline, không rơi về profile mặc định.

Mỗi level trong report có batch size, concurrency, item count, input
characters, elapsed time, throughput, status và error category. Level lỗi vẫn
được ghi để so sánh; recommendation chỉ chọn từ level hợp lệ.

## Artifacts và exit codes

Candidate, rerank score và metrics artifacts được publish thành bundle có
`manifest.json`. Candidate bundle dùng chung tại `candidates/`; rerank bundles
được lưu theo `rerank/<model-slug>/<variant-id>/`, còn reports theo
`reports/baseline/<metrics-id>/` hoặc
`reports/rerank/<model-slug>/<variant-id>/<metrics-id>/`.

Global corpus/query embeddings vẫn là cache JSONL dưới `data/heavy/cache/`. Rerank execution cache cũng
được version theo model SHA và prompt contract:

```text
data/heavy/cache/vector_embeddings/<model-slug>.jsonl
data/heavy/cache/query_embeddings/<model-slug>.jsonl
data/heavy/cache/rerank_scores/<model-slug>/<execution-identity>.jsonl
```

Mỗi record có checksum; cache resume theo key, còn run-owned bundle là nguồn
artifact chuẩn và không bị ghi đè bởi model khác.
Retrieval, rerank và metrics dùng chung workspace named qua `--run NAME`. Không trộn artifact candidate hoặc
input giữa các run/model/evaluation identity khác nhau.

| Code | Meaning |
| ---: | --- |
| 0 | Stage completed successfully |
| 1 | Runtime or validation failure |
| 2 | Invalid CLI usage |
| 3 | Resumable incomplete stage |
| 130 | Interrupted by the operator |
