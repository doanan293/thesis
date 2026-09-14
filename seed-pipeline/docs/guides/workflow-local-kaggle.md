# Local + Kaggle workflow

Build, import, retrieve và metrics chạy ở local; Kaggle chạy embedding (bundle và query) và reranker.

## 1. Kiểm tra môi trường

Khai báo profile Kaggle trong `seed-pipeline/.env`:

```dotenv
KAGGLE_ACCOUNT_DEFAULT=acc1
KAGGLE_SHARED_OWNER=account_goc
KAGGLE_ACC1_USERNAME=account_goc
KAGGLE_ACC1_API_TOKEN=token_acc1
KAGGLE_ACC2_USERNAME=account_phu_2
KAGGLE_ACC2_API_TOKEN=token_acc2
```

```bash
uv sync --directory ..    # ở repo root: một .venv chung cho uv workspace
uv run seed doctor --backend kaggle --kaggle-account acc1
docker compose up -d postgres qdrant llama-embedding   # tìm compose.yaml ở thư mục cha
```

Production stage ở profile mode tự tìm checkpoint tương thích trong mọi profile `accN` và mirror sang owner đích trước khi submit kernel.

## 2. Build, validate và export bundle

```bash
uv run seed build
uv run seed validate
uv run seed bundle export --output data/heavy/bundles/formulary --force
```

## 3. Embed bundle trên Kaggle

```bash
uv run seed bundle embed --bundle data/heavy/bundles/formulary --backend kaggle \
  --model qwen3-embedding:4b-fp16 --kaggle-account acc1 --dry-run
uv run seed bundle embed --bundle data/heavy/bundles/formulary --backend kaggle \
  --model qwen3-embedding:4b-fp16 --kaggle-account acc1
```

Lệnh stream log kernel, tải artifact về và merge vào `data/heavy/cache/text_embeddings/qwen3_embedding_4b_fp16.jsonl`. Exit code 3 nghĩa là kernel chưa xong hoặc hết budget: chạy lại đúng lệnh để resume từ checkpoint. Khi cache đủ, lệnh ghi `embeddings/qwen3_embedding_4b_fp16.jsonl` vào bundle. `Ctrl-C` chỉ dừng theo dõi; kernel vẫn chạy và lần chạy sau tự attach.

## 4. Import vào backend

```bash
cd ../backend
uv run pharma-agent migrate
uv run pharma-agent corpus import ../seed-pipeline/data/heavy/bundles/formulary --collection formulary --publish
uv run pharma-agent corpus releases --collection formulary
cd ../seed-pipeline
```

## 5. Evaluation dataset và retrieval

```bash
uv run seed evaluation build --bundle data/heavy/bundles/formulary
uv run seed embed queries --backend kaggle --model qwen3-embedding:4b-fp16 --kaggle-account acc1
uv run seed retrieve --run backend-bm25-k30 --retriever bm25 --candidate-k 30
uv run seed retrieve --run backend-hybrid-qwen4b-p50-k30-rrf2 --retriever hybrid \
  --prefetch-k 50 --candidate-k 30 --rrf-k 2
uv run seed metrics --run backend-hybrid-qwen4b-p50-k30-rrf2 --top-k 30
```

Thêm `--limit 50` và tên run `...-smoke50` để chạy thử trước.

## 6. Rerank trên Kaggle và metrics

```bash
uv run seed rerank --run backend-hybrid-qwen4b-p50-k30-rrf2 --backend kaggle \
  --model qwen3-reranker:4b-fp16 --kaggle-account acc2 --dry-run
uv run seed rerank --run backend-hybrid-qwen4b-p50-k30-rrf2 --backend kaggle \
  --model qwen3-reranker:4b-fp16 --kaggle-account acc2
uv run seed metrics --run backend-hybrid-qwen4b-p50-k30-rrf2 --model qwen3-reranker:4b-fp16 --top-k 30
```

Kaggle không truy cập Postgres/Qdrant và không tính metrics; rerank chỉ nhận candidate JSONL cùng manifest.

## Troubleshooting

- Credential/owner lỗi: sửa `.env`, chạy lại `seed doctor --backend kaggle`.
- `checksum mismatch` hoặc `no mounted file found`: dataset input chưa `READY` hoặc lệch; chạy lại để reconciler publish đúng version.
- `Expected hits from exactly one published release`: chưa import/publish release, hoặc `retrieval.collections` trỏ sai collection.
- `pinned to release`: release hiện hành đổi giữa chừng; dùng tên run mới.
- `Query embedding cache ... is missing`: chạy lại `seed embed queries --backend kaggle` với đúng file evaluation và model embedding của backend.
- `--top-k` lớn hơn `candidate-k`: retrieve lại với candidate depth đủ lớn.

Chi tiết lệnh: [CLI reference](cli-reference.md). Chính sách bàn giao: [Downstream](downstream.md).
