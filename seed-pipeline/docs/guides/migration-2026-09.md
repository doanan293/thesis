# Chuyển đổi 2026-09: corpus-pipeline → seed-pipeline và corpus platform

Runbook một lần cho môi trường dev theo spec `backend/docs/superpowers/specs/2026-09-13-corpus-platform-design.md` §11. Postgres và Qdrant dev bị xoá và dựng lại; không có backfill. Mọi lệnh `uv run seed` chạy trong `seed-pipeline/`, lệnh `pharma-agent` chạy trong `backend/`.

Cần sẵn: bản archive `data/heavy/` (raw PDF, snapshot An Khang, bộ gold `processed/evaluation/section_retrieval_eval.jsonl`) giải nén vào `seed-pipeline/data/heavy/`, `seed-pipeline/.env` có profile Kaggle, `backend/.env` có Postgres, Qdrant và embedding endpoint.

## 1. Backend

Backend đã có schema `corpus`, domain corpus, `pharma-agent corpus import`, layout Qdrant mới, `PostgresHydrator`, audit và health check (plan P1–P3 đã merge).

```bash
cd /home/andv/personal/thesis
uv sync
cd backend
uv run pytest -q
uv run pytest -q -m integration
```

## 2. seed-pipeline: đổi tên và export

```bash
cd /home/andv/personal/thesis
uv sync
cd seed-pipeline
uv run seed doctor --backend kaggle --kaggle-account acc1
```

Export chạy trong bước 3, trên cả build cũ lẫn build mới.

## 3. Kiểm tra khớp chunk (bắt buộc 100%)

Parity so `chunk_section` của backend với `chunks.jsonl` do code cũ sinh ra. Code cũ bị xoá ở commit `refactor(seed): remove the unified chunk and Qdrant upload contracts`, nên build cũ chạy trên commit ngay trước đó:

```bash
cd /home/andv/personal/thesis
PRE_REMOVAL=$(git log --format=%H -1 --grep="evaluate retrieval through the backend with cached query vectors")
git switch --detach "$PRE_REMOVAL"
cd seed-pipeline
uv sync
```

```bash
uv run seed build
uv run seed validate
mkdir -p data/heavy/migration
cp data/heavy/processed/rag-final/chunks.jsonl data/heavy/migration/rag-final-chunks.jsonl
uv run seed bundle export --output data/heavy/bundles/formulary --force
uv run seed bundle parity --bundle data/heavy/bundles/formulary --old-chunks data/heavy/migration/rag-final-chunks.jsonl --report data/heavy/migration/parity-pre-removal.json
uv run pytest -q -m data tests/bundle/test_parity_data.py
```

Kết quả phải là `mismatches=0` và test `data` pass. Nếu lệch, `parity-pre-removal.json` liệt kê section, ordinal, field, giá trị cũ và mới; sửa chunker trong backend (P1) hoặc quy tắc export (`seed_pipeline/bundle/export.py`) rồi chạy lại, không chuyển sang bước 4.

Quay lại nhánh làm việc, build lại theo contract `rag-final-v3` và kiểm tra lần nữa với cùng bản sao `chunks.jsonl`:

```bash
cd /home/andv/personal/thesis
git switch -
uv sync
cd seed-pipeline
```

```bash
uv run seed build
uv run seed validate
uv run seed bundle export --output data/heavy/bundles/formulary --force
uv run seed bundle parity --bundle data/heavy/bundles/formulary --old-chunks data/heavy/migration/rag-final-chunks.jsonl --report data/heavy/migration/parity-final.json
```

## 4. Embedding

```bash
uv run seed bundle embed --bundle data/heavy/bundles/formulary --backend kaggle --model qwen3-embedding:4b-fp16 --kaggle-account acc1 --dry-run
uv run seed bundle embed --bundle data/heavy/bundles/formulary --backend kaggle --model qwen3-embedding:4b-fp16 --kaggle-account acc1
```

Exit code 3 là kernel chưa xong: chạy lại đúng lệnh để resume. Khi xong, bundle có `embeddings/qwen3_embedding_4b_fp16.jsonl` và `manifest.json` liệt kê file này. `--model` phải trùng `PHARMA_RETRIEVAL__EMBEDDING__MODEL` của backend.

## 5. Reset dev và import

Dừng backend nếu đang chạy, xoá collection Qdrant cũ (alias đi cùng collection), dựng lại Postgres từ migration rồi import:

```bash
cd /home/andv/personal/thesis
docker compose stop backend
for name in $(curl -s http://localhost:6333/collections | python3 -c 'import json, sys; print("\n".join(c["name"] for c in json.load(sys.stdin)["result"]["collections"] if c["name"].startswith("thesis_chunks_")))'); do
  curl -s -X DELETE "http://localhost:6333/collections/$name"
done
docker compose rm -sf postgres
docker volume rm thesis_postgres_data
docker compose up -d postgres qdrant llama-embedding llama-reranker
```

```bash
cd /home/andv/personal/thesis/backend
uv run pharma-agent migrate
uv run pharma-agent corpus import ../seed-pipeline/data/heavy/bundles/formulary --collection formulary --publish
uv run pharma-agent corpus releases --collection formulary
uv run pharma-agent check
```

`corpus releases` phải cho một release `ready` là release hiện hành, số chunk bằng số chunk đã kiểm ở bước 3; import không gọi embedding endpoint cho chunk nào vì mọi vector có trong bundle.

## 6. Kiểm tra chất lượng

Dùng đúng bộ gold của run cũ để kết quả so được:

```bash
cd /home/andv/personal/thesis/seed-pipeline
sha256sum data/heavy/processed/evaluation/section_retrieval_eval.jsonl
```

Hash phải là `b26d9fd71a6dd28a65b571449b612c13510b7e1568bddbcfdb5156e21ac8b1a0` (ghi trong `data/retrieval_eval/hybrid-qwen4b-p50-k30-rrf2/run.json`). Không chạy lại `seed evaluation build` trong bước này.

```bash
uv run seed embed queries --backend kaggle --model qwen3-embedding:4b-fp16 --kaggle-account acc1
uv run seed retrieve --run backend-hybrid-qwen4b-p50-k30-rrf2-smoke50 --retriever hybrid --prefetch-k 50 --candidate-k 30 --rrf-k 2 --limit 50
uv run seed retrieve --run backend-hybrid-qwen4b-p50-k30-rrf2 --retriever hybrid --prefetch-k 50 --candidate-k 30 --rrf-k 2
uv run seed rerank --run backend-hybrid-qwen4b-p50-k30-rrf2 --backend kaggle --model qwen3-reranker:4b-fp16 --kaggle-account acc2
uv run seed metrics --run backend-hybrid-qwen4b-p50-k30-rrf2 --top-k 30
```

`seed retrieve` không embed query qua endpoint: vector lấy từ `data/heavy/cache/query_embeddings/qwen3_embedding_4b_fp16.jsonl` do `seed embed queries` ghi (resume theo checkpoint, chỉ query mới hoặc đổi text mới chạy inference). Khi cache không đổi, `query_embeddings_sha256` trong `run.json` của run mới bằng `d9769fab90929314e959fb7abb75286bcc1a6436a1607d1b18fff6480d716b42` của run cũ; thiếu query nào thì `seed retrieve` dừng với `Query embedding cache ... is missing`.

So report `data/retrieval_eval/backend-hybrid-qwen4b-p50-k30-rrf2/reports/rerank/qwen3_reranker_4b_fp16/*/*/report.md` với ngưỡng (giảm tối đa 1 điểm phần trăm; MRR tối đa 0.01):

| Chỉ số | `hybrid-qwen4b-p50-k30-rrf2` + `qwen3-reranker:4b-fp16` | Ngưỡng chấp nhận |
| --- | ---: | ---: |
| Hit@10 | 96.98% | ≥ 95.98% |
| MRR | 0.8060 | ≥ 0.7960 |

Tham khảo baseline không rerank (`reports/baseline/*/report.md`): Hit@10 95.67% (≥ 94.67%), MRR 0.7242 (≥ 0.7142). Nếu dùng reranker khác, lấy số của đúng reranker đó trong `data/retrieval_eval/hybrid-qwen4b-p50-k30-rrf2/reports/rerank/<model-slug>/`.

Không đạt ngưỡng thì dừng ở đây: kiểm tra `parity-final.json`, `corpus releases`, collection Qdrant `chunks_current` và so từng query giữa candidates cũ (`data/heavy/retrieval_eval/hybrid-qwen4b-p50-k30-rrf2/candidates/`) và mới theo `section_id`.

## 7. Dọn dẹp

Code và tài liệu cũ đã bị xoá ở commit `refactor(seed): remove the unified chunk and Qdrant upload contracts`. Sau khi bước 6 đạt:

```bash
cd /home/andv/personal/thesis/seed-pipeline
rm -rf data/heavy/cache/vector_embeddings data/heavy/.work/kaggle-chunk-embeddings
rm -rf data/heavy/migration
cd /home/andv/personal/thesis
grep -n "COLLECTION_ALIAS" compose.yaml backend/.env.example
```

`grep` không được in gì (`compose.yaml` không đặt alias; backend dùng `retrieval.qdrant_collection = "chunks_current"`). Corpus lúc chạy từ nay nạp bằng `pharma-agent corpus import` như bước 5; ghi Hit@10 và MRR đạt được vào mô tả PR của đợt chuyển đổi.
