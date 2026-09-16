# pharma-lab

Công cụ offline tạo dữ liệu gốc cho backend `pharma-agent` từ nguồn nội bộ: **Dược thư Quốc gia Việt Nam** (PDF) và tờ hướng dẫn sử dụng thuốc (HTML đã crawl). Đầu ra là một knowledge bundle `knowledge-bundle/v1`; backend nạp bundle bằng `pharma-agent corpus import`. pharma-lab không ghi thẳng vào Postgres hay Qdrant.

## Bắt đầu

```bash
cd /home/andv/personal/thesis
uv sync    # uv workspace: backend `pharma-agent` và mọi thư viện dev của pharma-lab
cd pharma-lab
```

| Trường hợp | Workflow |
| --- | --- |
| Build ở local, Kaggle GPU embed bundle và chấm rerank | [Local + Kaggle GPU](docs/guides/workflow-local-kaggle.md) |
| Không dùng Kaggle; mọi model chạy local | [Local CPU-only](docs/guides/workflow-local-only.md) |
| Đánh giá retrieval (dựng lại số liệu luận văn) | [Evaluation](docs/guides/evaluation.md) |
| Đánh giá end-to-end agent trên bộ golden | [E2E evaluation](docs/guides/e2e-evaluation.md) |

## Luồng dữ liệu

```text
pharma-lab source crawl      sitemap -> data/sources/leaflets/ (html, danh sách URL, manifest)
pharma-lab build             PDF + HTML tờ hướng dẫn -> data/corpus/rag-final/ (sections + blocks)
pharma-lab validate          kiểm tra contract rag-final-v3
pharma-lab bundle export     rag-final -> data/corpus/formulary/ (knowledge-bundle/v1)
pharma-lab bundle embed      chunk_section của backend -> embeddings/<model_slug>.jsonl (Kaggle hoặc local)
pharma-agent corpus import ../pharma-lab/data/corpus/formulary --collection formulary --publish   (chạy trong backend/)
pharma-lab evaluation build  bundle -> bộ câu hỏi gold data/evaluation/gold/ (section key)
pharma-lab embed queries     bộ gold -> cache vector query (Kaggle hoặc local)
pharma-lab retrieve          RetrievalService của backend -> data/evaluation/runs/<run>/candidates/
pharma-lab rerank            chấm reranker trên candidates của run -> rerank/<model>/ (local hoặc Kaggle)
pharma-lab metrics           Hit@K, MRR -> reports/baseline/ hoặc reports/rerank/<model>/, thư mục top<K>-window<N>/
```

## Mental model cho retrieval evaluation

- `--run` là tên workspace/thí nghiệm dùng chung cho retrieve, rerank và metrics; tên này không chọn thuật toán.
- `pharma-lab retrieve` gọi `build_retrieval_service(settings, embedder=...)` của backend trên Postgres và Qdrant dev đã import và publish release. Settings đọc từ `../backend/.env` (`--backend-env-file`). `--retriever` là `bm25`, `dense` hoặc `hybrid`; `--prefetch-k`, `--candidate-k`, `--rrf-k` được đưa vào cấu hình retrieval của backend. Với `dense` và `hybrid`, vector query lấy từ cache của `pharma-lab embed queries` (`--query-embeddings`); thiếu query nào thì lệnh dừng và báo, không embed lại qua endpoint.
- Run identity ghi `release_id`, `chunker_version` và `embedding_model`; release khác thì dùng tên run mới.
- `--model` của `pharma-lab rerank` là reranker model; `pharma-lab metrics` không chạy model.
- Mỗi run nằm trong một thư mục `data/evaluation/runs/<run>/`; `run.json` ghi identity và `origin` (`backend` hoặc `imported`). Mở lại run với input khác thì lệnh dừng và nêu các trường khác nhau; `--force` thay thế.

## Prerequisites chung

- `uv`, Docker và Docker Compose.
- PDF Dược thư, HTML tờ hướng dẫn và các file nguồn nhỏ (mappings, glossary, curated tables, danh sách âm tiết) dưới `data/sources/`; lần crawl đầu cần `pharma-lab source crawl --sitemap-url`.
- GGUF dưới `../ai-models/gguf/`.
- `backend/.env` cho `pharma-lab retrieve` và `pharma-agent corpus import`; `.env` của pharma-lab cho Kaggle.

## Đầu ra

- `data/corpus/rag-final/`: `sections.jsonl`, `blocks.jsonl`, `manifest.json`, `validation_report.json` (contract `rag-final-v3`).
- `data/corpus/formulary/`: knowledge bundle bàn giao cho backend.
- `data/cache/text_embeddings/<model>.jsonl`: cache embedding theo `sha256(embedding_text)`, dùng lại giữa các lần build.
- `data/cache/query_embeddings/<model>.jsonl`, `data/cache/rerank_scores/<model>.jsonl`, `data/cache/kaggle_profiles/<workload>/<model>.json`, `data/cache/local_profiles/<workload>/<model>.json`: cache của `pharma-lab embed queries`, `pharma-lab rerank`, profile runtime Kaggle và profile reranker CPU local.
- `data/evaluation/runs/<run>/`: `run.json`, `candidates/`, `rerank/<model>/`, `reports/`.

Git chỉ giữ file nhỏ; phần còn lại nằm trong archive Kaggle, xem [data/README.md](data/README.md).

Chính sách bàn giao và artifact: [Downstream](docs/guides/downstream.md).

## Cấu trúc project

- `src/pharma_lab/orchestration/`: build và atomic publish `rag-final`.
- `src/pharma_lab/corpus/`: trích xuất PDF, crawl tờ hướng dẫn, canonical blocks, bảng Docling, validation.
- `src/pharma_lab/bundle/`: export, embed và evaluation chunks dựa trên `pharma_agent.domain.corpus`.
- `src/pharma_lab/embeddings/`: cache embedding theo hash và backend local/Kaggle.
- `src/pharma_lab/integrations/kaggle/`: reconciliation, checkpoint và worker trên Kaggle (worker không import backend).
- `src/pharma_lab/evaluation/`: dataset, retrieval qua backend, rerank, metrics.
- `src/pharma_lab/cli/`: entrypoint `uv run pharma-lab`.

## CLI

Entrypoint duy nhất là `uv run pharma-lab`. Xem [CLI reference](docs/guides/cli-reference.md); `uv run pharma-lab COMMAND --help` cho default thực tế.

## Kiểm tra

```bash
uv run ruff check src tests && uv run ruff format --check src tests
uv run pyrefly check --min-severity warn
uv run pytest -q
```
