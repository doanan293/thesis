# seed-pipeline

Công cụ offline tạo dữ liệu gốc cho backend `pharma-agent` từ nguồn nội bộ: **Dược thư Quốc gia Việt Nam** (PDF) và snapshot tờ hướng dẫn sử dụng **An Khang**. Đầu ra là một knowledge bundle `knowledge-bundle/v1`; backend nạp bundle bằng `pharma-agent corpus import`. seed-pipeline không ghi thẳng vào Postgres hay Qdrant.

## Bắt đầu

```bash
cd /home/andv/personal/thesis/seed-pipeline
uv sync    # cài luôn backend `pharma-agent` (path dependency, editable)
```

| Trường hợp | Workflow |
| --- | --- |
| Build ở local, Kaggle GPU embed bundle và chấm rerank | [Local + Kaggle GPU](docs/guides/workflow-local-kaggle.md) |
| Không dùng Kaggle; mọi model chạy local | [Local CPU-only](docs/guides/workflow-local-only.md) |
| Chuyển từ corpus-pipeline cũ (một lần, môi trường dev) | [Migration 2026-09](docs/guides/migration-2026-09.md) |

## Luồng dữ liệu

```text
seed build             PDF + snapshot An Khang -> data/heavy/processed/rag-final/ (sections + blocks)
seed validate          kiểm tra contract rag-final-v3
seed bundle export     rag-final -> data/heavy/bundles/formulary/ (knowledge-bundle/v1)
seed bundle embed      chunk_section của backend -> embeddings/<model_slug>.jsonl (Kaggle hoặc local)
pharma-agent corpus import ../seed-pipeline/data/heavy/bundles/formulary --collection formulary --publish   (chạy trong backend/)
seed evaluation build  bundle -> bộ câu hỏi gold (section key)
seed embed queries     bộ gold -> cache vector query (Kaggle hoặc local)
seed retrieve          RetrievalService của backend -> candidate bundle của run
seed rerank            chấm reranker trên candidate bundle (local hoặc Kaggle)
seed metrics           Hit@K, MRR từ artifact đã đóng băng
```

## Mental model cho retrieval evaluation

- `--run` là tên workspace/thí nghiệm dùng chung cho retrieve, rerank và metrics; tên này không chọn thuật toán.
- `seed retrieve` gọi `build_retrieval_service(settings, embedder=...)` của backend trên Postgres và Qdrant dev đã import và publish release. Settings đọc từ `../backend/.env` (`--backend-env-file`). `--retriever` là `bm25`, `dense` hoặc `hybrid`; `--prefetch-k`, `--candidate-k`, `--rrf-k` được đưa vào cấu hình retrieval của backend. Với `dense` và `hybrid`, vector query lấy từ cache của `seed embed queries` (`--query-embeddings`); thiếu query nào thì lệnh dừng và báo, không embed lại qua endpoint.
- Run identity ghi `release_id`, `chunker_version` và `embedding_model`; release khác thì dùng tên run mới.
- `--model` của `seed rerank` là reranker model; `seed metrics` không chạy model.

### Re-judge metrics trên artifacts hiện có

Khi chỉ sửa judgment mà không đổi query text, có thể chạy lại evaluation mà không retrieve hoặc rerank lại. Lệnh mặc định là dry-run:

```bash
uv run seed evaluation rejudge-current \
  --dense-run dense-qwen4b-k30 \
  --hybrid-run hybrid-qwen4b-p50-k30-rrf2
```

Thêm `--apply` để thay evaluation dataset, cập nhật hash, xoá metrics reports cũ của hai run và tạo lại baseline cùng rerank reports từ candidates/rerank scores đang có.

## Prerequisites chung

- `uv`, Docker và Docker Compose.
- Raw PDF và snapshot An Khang dưới `data/heavy/raw/`; mappings, glossary, curated tables, URL lists và danh sách âm tiết dưới `data/resources/`; source manifests dưới `data/manifests/source/`.
- GGUF dưới `../ai-models/gguf/`.
- `backend/.env` cho `seed retrieve` và `pharma-agent corpus import`; `.env` của seed-pipeline cho Kaggle.

## Đầu ra

- `data/heavy/processed/rag-final/`: `sections.jsonl`, `blocks.jsonl`, `manifest.json`, `validation_report.json` (contract `rag-final-v3`); bản sao manifest nhỏ ở `data/manifests/corpus/`.
- `data/heavy/bundles/formulary/`: knowledge bundle bàn giao cho backend.
- `data/heavy/cache/text_embeddings/<model-slug>.jsonl`: cache embedding theo `sha256(embedding_text)`, dùng lại giữa các lần build.
- `data/heavy/cache/query_embeddings/`, `data/heavy/cache/rerank_scores/`: cache của `seed embed queries` và `seed rerank`.
- `data/retrieval_eval/<run>/` và `data/heavy/retrieval_eval/<run>/`: run metadata, candidates, rerank variants, reports.

Chính sách bàn giao và artifact: [Downstream](docs/guides/downstream.md).

## Cấu trúc project

- `src/seed_pipeline/orchestration/`: build và atomic publish `rag-final`.
- `src/seed_pipeline/corpus/`: trích xuất PDF, crawl An Khang, canonical blocks, bảng Docling, validation.
- `src/seed_pipeline/bundle/`: export, parity, embed và evaluation chunks dựa trên `pharma_agent.domain.corpus`.
- `src/seed_pipeline/embeddings/`: cache embedding theo hash và backend local/Kaggle.
- `src/seed_pipeline/integrations/kaggle/`: reconciliation, checkpoint và worker trên Kaggle (worker không import backend).
- `src/seed_pipeline/evaluation/`: dataset, retrieval qua backend, rerank, metrics.
- `src/seed_pipeline/cli/`: entrypoint `uv run seed`.

## CLI

Entrypoint duy nhất là `uv run seed`. Xem [CLI reference](docs/guides/cli-reference.md); `uv run seed COMMAND --help` cho default thực tế.

## Kiểm tra

```bash
uv run ruff check src tests && uv run ruff format --check src tests
uv run pyrefly check --min-severity warn
uv run pytest -q
uv run pytest -q -m data     # cần build thật dưới data/heavy (xem migration guide)
```
