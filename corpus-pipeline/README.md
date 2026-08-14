# Pipeline Xử Lý Dữ Liệu Hybrid PyMuPDF + Docling

Pipeline chuẩn bị và tối ưu dữ liệu từ **Dược thư Quốc gia Việt Nam** cùng
snapshot **An Khang** cho hệ thống Retrieval-Augmented Generation (RAG) và AI
Agent y dược.

## Bắt đầu

Mọi lệnh được chạy từ project root:

```bash
cd /home/andv/personal/thesis/corpus-pipeline
uv sync
```

Chọn đúng một workflow:

| Trường hợp | Workflow |
| --- | --- |
| Máy local build corpus, vận hành Qdrant và tính metrics; Kaggle GPU chạy embedding/reranking | [Local + Kaggle GPU](docs/guides/workflow-local-kaggle.md) |
| Không sử dụng Kaggle; toàn bộ model chạy local CPU | [Local CPU-only](docs/guides/workflow-local-only.md) |

Hai workflow đều bắt đầu từ raw inputs đã có trong `data/raw/`, chạy smoke test
50 queries, sau đó mới mở rộng lên benchmark đầy đủ 10.000 queries. Workflow
Local + Kaggle GPU là đường chạy chính khi cần embedding/reranking nhanh hơn.

## Mental model cho retrieval evaluation

- `--run` là tên workspace/thí nghiệm dùng chung cho retrieval, rerank và
  metrics; tên này không chọn thuật toán.
- `--retriever` chọn `bm25`, `dense` hoặc `hybrid`.
- `--model` của `corpus retrieve` chọn embedding model và Qdrant collection;
  workflow baseline dùng `qwen3-embedding:0.6b-fp16`.
- `--model` của `corpus rerank` chọn reranker model; baseline dùng
  `qwen3-reranker:0.6b-fp16`. `corpus metrics` không chạy model.

Ví dụ, run `dense-qwen06b-k30` là tên thí nghiệm, còn `--retriever dense` mới
là lựa chọn thuật toán. Xem [CLI reference](docs/guides/cli-reference.md) để biết
run identity và quy ước đặt tên đầy đủ.

## Prerequisites chung

- Python environment được quản lý bằng `uv`.
- Docker và Docker Compose hoạt động.
- Raw PDF, An Khang snapshot, curated tables, mappings, glossary và danh sách âm
  tiết đã có dưới `data/raw/`.
- GGUF baseline đã có dưới `../ai-models/gguf/`.
- Workflow Kaggle cần credentials/owner hợp lệ trong `.env`.

Các guide có preflight command cụ thể để kiểm tra từng input trước khi chạy.

## Đầu ra cuối

- `data/processed/rag-final/`: final sections, chunks, manifest và validation
  report.
- `data/processed/evaluation/`: evaluation dataset 10.000 queries.
- `data/cache/`: các cache JSONL phẳng có thể resume/reuse và tự kiểm tra
  checksum (`vector_embeddings/<model>.jsonl`,
  `query_embeddings/<model>.jsonl`, `rerank_scores/<model>.jsonl`).
- `data/runs/retrieval_eval/`: per-query checkpoint và Markdown metrics report.
- Qdrant collection theo embedding model, ví dụ
  `thesis_chunks_qwen3_embedding_0_6b_fp16`.

Sau khi pipeline hoàn tất, áp dụng
[Downstream và Data Artifact Policy](docs/guides/downstream.md) khi tích hợp
RAG/Agent.

## Cấu trúc project

- `src/corpus_pipeline/orchestration/`: Build và atomic publish final corpus.
- `src/corpus_pipeline/corpus/`: Extraction, crawling, canonical blocks, tables và metadata.
- `src/corpus_pipeline/vector_store/`: Embedding cache và Qdrant ingest.
- `src/corpus_pipeline/integrations/kaggle/`: Kaggle reconciliation, checkpoint và workers.
- `src/corpus_pipeline/evaluation/`: Dataset generation, retrieval, reranking và metrics.
- `src/corpus_pipeline/corpus/validation/`: Final corpus quality gates.
- `src/corpus_pipeline/cli/`: Public typed command-line entrypoint.
- `tests/`: Unit và integration contract tests.

## CLI

Public command surface duy nhất là typed entrypoint `uv run corpus`. Xem
[CLI reference](docs/guides/cli-reference.md) để biết các stage và shared flags;
dùng `uv run corpus COMMAND --help` để xem default thực tế.

## Kiểm tra hệ thống

```bash
uv run pytest -q
```
