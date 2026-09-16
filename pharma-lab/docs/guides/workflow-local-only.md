# Local-only workflow

Mọi model chạy local qua llama.cpp trong `../compose.yaml`; không dùng Kaggle.

## 1. Kiểm tra môi trường

```bash
uv sync --directory ..    # ở repo root: một .venv chung cho uv workspace
uv run pharma-lab doctor --backend local
docker compose up -d postgres qdrant llama-embedding llama-reranker   # tìm compose.yaml ở thư mục cha
```

## 2. Build, validate, export và embed

```bash
uv run pharma-lab build
uv run pharma-lab validate
uv run pharma-lab bundle export --output data/corpus/formulary --force
uv run pharma-lab bundle embed --bundle data/corpus/formulary --backend local --model qwen3-embedding:4b-fp16
```

Embed local chạy trên CPU nên chậm; lệnh ghi cache sau mỗi lô, chạy lại để tiếp tục. Có thể bỏ bước embed: `pharma-agent corpus import` tự embed phần thiếu qua endpoint của backend.

## 3. Import vào backend

```bash
cd ../backend
uv run pharma-agent migrate
uv run pharma-agent corpus import ../pharma-lab/data/corpus/formulary --collection formulary --publish
cd ../pharma-lab
```

## 4. Evaluation

```bash
uv run pharma-lab evaluation build --bundle data/corpus/formulary
uv run pharma-lab embed queries --backend local --model qwen3-embedding:4b-fp16
uv run pharma-lab retrieve --run backend-bm25-k30 --retriever bm25 --candidate-k 30
uv run pharma-lab metrics --run backend-bm25-k30 --top-k 30
uv run pharma-lab retrieve --run backend-dense-qwen4b-k30 --retriever dense --candidate-k 30
uv run pharma-lab metrics --run backend-dense-qwen4b-k30 --top-k 30
uv run pharma-lab retrieve --run backend-hybrid-qwen4b-p50-k30-rrf2 --retriever hybrid \
  --prefetch-k 50 --candidate-k 30 --rrf-k 2
uv run pharma-lab rerank --run backend-hybrid-qwen4b-p50-k30-rrf2 --backend local --benchmark --model qwen3-reranker:4b-fp16
uv run pharma-lab rerank --run backend-hybrid-qwen4b-p50-k30-rrf2 --backend local --model qwen3-reranker:4b-fp16
uv run pharma-lab metrics --run backend-hybrid-qwen4b-p50-k30-rrf2 --top-k 30
```

`--benchmark` chọn cấu hình `llama-reranker` có p95 thấp nhất trên máy này và lưu profile ở `data/cache/local_profiles/rerank/`; lệnh rerank sau đó dựng service theo profile và gửi một request `/v1/rerank` cho mỗi câu hỏi. Benchmark và rerank cả run chạy lâu, nên chạy trong tmux.

Mỗi baseline cần ít nhất 30 candidates/query để tính Hit@30. Thêm `--limit 50` với tên run riêng để smoke test; bỏ `--limit` làm đổi run identity.

Rerank trên CPU có thể chạy nhiều giờ: chạy lệnh trong một cửa sổ tmux riêng (`tmux new-session -s rerank`). Mỗi lệnh `pharma-lab rerank` ghi dòng bắt đầu, kết quả và lỗi vào `data/work/logs/rerank/<model>.log`; file này còn sau khi máy khởi động lại.

## Troubleshooting

- `Expected hits from exactly one published release`: import và publish bundle trước.
- `Query embedding cache ... is missing`: chạy `pharma-lab embed queries --backend local` với đúng file evaluation và model embedding của backend.
- Run identity xung đột: chọn `--run` mới thay vì ghi đè artifact cũ.

Chi tiết lệnh: [CLI reference](cli-reference.md). Chính sách bàn giao: [Downstream](downstream.md).
