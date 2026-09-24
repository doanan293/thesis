# Downstream và Data Artifact Policy

## 1. Bàn giao cho backend

1. Đầu ra bàn giao duy nhất của pharma-lab là knowledge bundle `knowledge-bundle/v1` do `pharma-lab bundle export` tạo và `pharma-lab bundle embed` bổ sung vector: `manifest.json`, `documents.jsonl`, `sections.jsonl`, `glossary.json`, `colloquial_mappings.json`, `embeddings/<model_slug>.jsonl`. Định dạng và quy tắc validate nằm trong model `pharma_agent.domain.corpus.bundle`.
2. Document key: `drug:<slug>`, `general:<slug>`, `leaflet:<category>:<slug>`. Tờ hướng dẫn sử dụng có `source.title` "Tờ hướng dẫn sử dụng" và không có `source.url`; nguồn gốc trang chỉ ghi trong `data/sources/leaflets/manifest.json`, không vào bundle. Section key trùng `section_id` của rag-final để bộ gold evaluation dùng tiếp.
3. Block `kind` (`prose`, `table`, `list`, `index_entries`) và section `retrieval` (`default`, `index_only`) thay cho các ID viết cứng trước đây (`BRAND_INDEX_SECTION_ID`, `APPENDIX_LIST_SECTION_IDS`).
4. Chia chunk, `context_header`, `embedding_text`, thuật ngữ và colloquial mapping do `chunk_section` của backend tính. pharma-lab import backend như thư viện và không có chunker riêng.
5. Vector được tính trước theo `(model, sha256(embedding_text))`. Backend chỉ dùng vector có model và số chiều khớp setting; phần thiếu backend tự embed qua endpoint đã cấu hình. Kernel Kaggle chỉ nhận text và hash, không import backend.
6. Chạy `uv run pharma-lab validate` trước `pharma-lab bundle export`. Export ghi bundle vào thư mục tạm, validate bằng `read_bundle`, rồi mới thay thư mục đích. Embed ghi vector của một model theo từng dòng vào file tạm (`write_bundle_embeddings`), chỉ thay `embeddings/<model_slug>.jsonl` và manifest khi mọi dòng hợp lệ, không đọc lại file của model khác. Backend import cũng chỉ nạp vector của model đang cấu hình (`read_bundle_embeddings`), nên RAM không tăng theo số model trong bundle.
7. Hydrate (`full_section`, `chunk_window`, `search_only`) là policy của backend (`pharma_agent.domain.corpus.hydrate`), không nằm trong bundle.
8. Đổi thuật toán chunk trong backend thì backend tăng `CHUNKER_VERSION`; import lại bundle tạo release mới.
9. Evaluation retrieval chạy qua `RetrievalService` của backend nên đo đúng retrieval lúc chạy; vector query lấy từ cache của `pharma-lab embed queries`, không embed lại; gold label vẫn theo section key.
10. Corpus lúc chạy nằm trong schema `corpus` của backend (Postgres là nguồn chính, Qdrant là index dẫn xuất), nạp bằng `pharma-agent corpus import ../pharma-lab/data/corpus/formulary --collection formulary --publish`. pharma-lab không ghi thẳng vào Postgres hay Qdrant.

## 2. Data Artifact Policy

- **Source inputs**: `data/sources/` gồm PDF Dược thư, `leaflets/` (HTML, danh sách URL, manifest), `curation/`, mappings, glossary và danh sách âm tiết.
- **Rebuildable workspace**: `data/work/build/in-progress/` trong lúc build (tự xoá sau publish; build lỗi giữ ở `data/work/build/failed/latest/`), input embedding `data/work/bundle-embed/`, chunk evaluation `data/work/evaluation-chunks/`, lock `data/work/locks/`.
- **Final reproducible contracts**: `data/corpus/rag-final/` (contract `rag-final-v3`) và bundle `data/corpus/formulary/`.
- **Caches**: `data/cache/text_embeddings/`, `query_embeddings/`, `rerank_scores/` (mỗi model một file `<model>.jsonl`, mỗi record có checksum), `kaggle_profiles/<workload>/<model>.json` và `local_profiles/<workload>/<model>.json`.
- **Experiment outputs**: bộ gold `data/evaluation/gold/` và run `data/evaluation/runs/<run>/`.
- Tên file và thư mục không chứa hash; digest nằm trong manifest và record. Git chỉ giữ file nhỏ (manifest, `run.json`, `report.md`, file nguồn tự duy trì); phần còn lại nằm trong archive Kaggle, xem `data/README.md`.
