# Downstream và Data Artifact Policy

## 1. Bàn giao cho backend

1. Đầu ra bàn giao duy nhất của seed-pipeline là knowledge bundle `knowledge-bundle/v1` do `seed bundle export` tạo và `seed bundle embed` bổ sung vector: `manifest.json`, `documents.jsonl`, `sections.jsonl`, `glossary.json`, `colloquial_mappings.json`, `embeddings/<model_slug>.jsonl`. Định dạng và quy tắc validate: spec `backend/docs/superpowers/specs/2026-09-13-corpus-platform-design.md` §5 và model `pharma_agent.domain.corpus.bundle`.
2. Document key: `drug:<slug>`, `general:<slug>`, `leaflet:ankhang:<category>:<slug>` (`source.url` của tờ hướng dẫn trỏ về `https://www.nhathuocankhang.com/<category>/<slug>`). Section key giữ nguyên `section_id` cũ để bộ gold evaluation dùng tiếp.
3. Block `kind` (`prose`, `table`, `list`, `index_entries`) và section `retrieval` (`default`, `index_only`) thay cho các ID viết cứng trước đây (`BRAND_INDEX_SECTION_ID`, `APPENDIX_LIST_SECTION_IDS`).
4. Chia chunk, `context_header`, `embedding_text`, thuật ngữ và colloquial mapping do `chunk_section` của backend tính. seed-pipeline import backend như thư viện và không có chunker riêng.
5. Vector được tính trước theo `(model, sha256(embedding_text))`. Backend chỉ dùng vector có model và số chiều khớp setting; phần thiếu backend tự embed qua endpoint đã cấu hình. Kernel Kaggle chỉ nhận text và hash, không import backend.
6. Chạy `uv run seed validate` trước `seed bundle export`. Export và embed ghi bundle vào thư mục tạm, validate bằng `read_bundle`, rồi mới thay thư mục đích.
7. Hydrate (`full_section`, `chunk_window`, `search_only`) là policy của backend (`pharma_agent.domain.corpus.hydrate`), không nằm trong bundle.
8. Đổi thuật toán chunk trong backend thì backend tăng `CHUNKER_VERSION`; import lại bundle tạo release mới.
9. Evaluation retrieval chạy qua `RetrievalService` của backend nên đo đúng retrieval lúc chạy; vector query lấy từ cache của `seed embed queries`, không embed lại; gold label vẫn theo section key.
10. Corpus lúc chạy nằm trong schema `corpus` của backend (Postgres là nguồn chính, Qdrant là index dẫn xuất), nạp bằng `pharma-agent corpus import <bundle_dir> --collection formulary --publish`. seed-pipeline không ghi thẳng vào Postgres hay Qdrant.

## 2. Data Artifact Policy

- **Source / provenance inputs**: inputs nhỏ trong `data/resources/`, source manifests trong `data/manifests/source/`, PDF và snapshot lớn trong `data/heavy/raw/`.
- **Rebuildable workspace**: `data/heavy/.work/` trong lúc build (tự xoá sau publish; workspace lỗi giữ ở `.work/failed/`), cùng input embedding `data/heavy/.work/bundle-embed/` và chunk evaluation `data/heavy/.work/evaluation-chunks/`.
- **Final reproducible contracts**: `data/heavy/processed/rag-final/` (contract `rag-final-v3`), snapshot manifest trong `data/manifests/corpus/`, bundle trong `data/heavy/bundles/`.
- **Caches**: `data/heavy/cache/text_embeddings/`, `query_embeddings/`, `rerank_scores/`; mỗi record có checksum.
- **Local experiment outputs**: metadata/report nhỏ trong `data/retrieval_eval/`; candidates, rerank bundles và per-query reports trong `data/heavy/retrieval_eval/`.
