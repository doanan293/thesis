# Hướng Dẫn Đánh Giá Retrieval (Evaluation)

Tài liệu này chi tiết quy trình tạo bộ dữ liệu đánh giá (ground truth CSV) và thực hiện các thử nghiệm retrieval với các mô hình embedding và reranker khác nhau.

---

## 1. Tạo CSV Đánh Giá Retrieval

Bộ evaluation dùng ground truth ở mức `section_id`: search chunk trước, sau đó hydrate theo section.

`cli.build_section_retrieval_eval` đọc `sections.jsonl` và `chunks.jsonl` để tạo ground truth offline.

### 1.1 Sinh Patient Natural Queries

```bash
uv run python -m evaluation.patient_query_generation
```

File `data/processed/evaluation/patient_queries.json` được sinh ra với `expected_section_id` thuộc `brand:ankhang:*`.

### 1.2 Sinh CSV Dataset Đánh Giá

```bash
uv run python -m cli.build_section_retrieval_eval
```

Output: `data/processed/evaluation/section_retrieval_eval.csv`

File CSV mặc định chứa 10,000 queries dùng chung cho mọi mô hình embedding, bao gồm các cột:
- `eval_group`: `formulary`, `ankhang`, `patient_natural`, `chunk_risk`, `multi_intent`, `noisy_confuser`.
- `source_family`: `drug_formulary`, `ankhang_brand`, `general_appendix`, `general_guidance`.
- `query_form`, `intent_category`, `source_subcategory`.
- `expected_section_ids`, `query_intent_count`, `answer_mode` (`single`, `multi_required`, `any_acceptable`).
- `retrieval_granularity` (`section`, `chunk_window`, `chunk_exact`, `multi_section`).

Metric đánh giá gồm: `hit@k`, `MRR`, multi-section recall/all-hit và breakdown theo từng phân nhóm.

---

## 2. Chạy Thử Nghiệm Retrieval Experiment

Runner hỗ trợ `dense`, `bm25` và `hybrid`, có resume theo JSONL checkpoint.

### 2.1 Dense Retrieval

```bash
uv run python -m evaluation.run_retrieval_eval \
  --retriever dense \
  --model qwen3-embedding:0.6b-fp16 \
  --top-k 10
```

### 2.2 BM25-Only Retrieval

```bash
uv run python -m evaluation.run_retrieval_eval \
  --retriever bm25 \
  --model qwen3-embedding:0.6b-fp16 \
  --top-k 10
```

### 2.3 Hybrid Dense + BM25

```bash
uv run python -m evaluation.run_retrieval_eval \
  --retriever hybrid \
  --model qwen3-embedding:0.6b-fp16 \
  --rrf-k 60 \
  --top-k 10
```

### 2.4 Hybrid Với Local Reranker

```bash
uv run python -m evaluation.run_retrieval_eval \
  --retriever hybrid \
  --model qwen3-embedding:0.6b-fp16 \
  --reranker qwen3-reranker:0.6b-fp16 \
  --candidate-k 50 \
  --top-k 10
```

Các reranker hợp lệ:
- `qwen3-reranker:0.6b-fp16`
- `qwen3-reranker:4b-fp16`
- `qwen3-reranker:8b-fp16`
- `bge-reranker-v2-m3:f16`
- `bge-reranker-v2-gemma:f16`

### 2.5 Query Embedding Cache & Endpoint Nâng Cao

Query embedding cache tự động lưu tại `data/cache/query_embeddings/`.

Ví dụ chạy với giới hạn mẫu và retry policy:
```bash
uv run python -m evaluation.run_retrieval_eval \
  --retriever hybrid \
  --model embeddinggemma:300m \
  --limit 20 \
  --candidate-k 30 \
  --top-k 10 \
  --query-input-batch-size 1 \
  --eval-max-retries 3 \
  --eval-retry-base-sleep 5
```

Ví dụ dùng llama.cpp server external cho evaluation:
```bash
uv run python -m evaluation.run_retrieval_eval \
  --retriever hybrid \
  --model embeddinggemma:300m \
  --reranker bge-reranker-v2-m3:f16 \
  --server-mode external \
  --llama-server-url http://127.0.0.1:8080 \
  --reranker-server-url http://127.0.0.1:8081
```

Output mặc định được lưu dưới `data/runs/retrieval_eval/` bao gồm JSONL per-query và Markdown summary.
