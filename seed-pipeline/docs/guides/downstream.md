# Hướng Dẫn Vận Hành Downstream & Data Artifact Policy

Tài liệu này chi tiết các quy tắc vận hành tích hợp cho downstream RAG/Agent và chính sách quản lý artifact dữ liệu của dự án.

Cả hai workflow Local + Kaggle GPU và Local CPU-only đều bàn giao vào policy
chung này sau khi corpus, Qdrant collection và evaluation report đã hoàn tất.

---

## 1. Vận Hành Downstream Agent

1. `full.md` là provenance text từ PDF qua PyMuPDF.
2. Docling table extraction toàn PDF mất nhiều thời gian; nếu `tables.curated.jsonl` còn tốt thì không cần chạy lại bước extract bảng.
3. Production runtime dùng Qdrant để search dense hoặc hybrid dense + BM25 sparse. Qdrant trả chunk payload có `chunk_id`, `section_id`, `chunk_index`, `hydrate_strategy`, `source`, `title`, `section`, `start_page`, `end_page`, `context_header`, `payload.chunk_text`, `colloquial_mapping`, `term_annotations`, và `payload.embedding_text` để rerank/hydrate tại chỗ.
4. Khi hydrate context, dùng `hydrate_strategy`:
   - `full_section`: Fetch tất cả Qdrant points cùng `section_id` và sort theo `chunk_index`.
   - `chunk_window`: Fetch chunk hit cộng một số chunk lân cận theo `section_id`/`chunk_index`.
   - `search_only`: Dùng `payload.chunk_text` như evidence ngắn, không hydrate nguyên section.
5. Các bảng được gắn vào section qua `table_ids`; section có bảng sẽ có `source_mix` gồm `pymupdf_text` và `docling_table`.
6. Table chunk được embed/search như chunk thường; khi hit vào bảng, downstream vẫn hydrate theo `section_id` và có thể dùng `table_id` để ưu tiên hiển thị hoặc trích riêng bảng liên quan.
7. Với `chunk_window`, policy downstream nên lấy chunk hit cộng 1-2 chunk trước/sau trong cùng `section_id` từ Qdrant nếu cần mở rộng context, tùy ngân sách token.
8. Với `search_only`, không hydrate nguyên section. Trường hợp cần tra cứu biệt dược thì dùng chunk hit trong Qdrant làm evidence hoặc lookup theo dòng/index riêng.
9. Luôn chạy `uv run seed validate` trước khi ingest corpus mới vào vector database.
10. Postgres app chỉ dùng cho trạng thái ứng dụng như user/session/message/retrieval audit/feedback qua `src/seed_pipeline/integrations/postgres/schema/rag_app_schema.sql`; corpus text không được import vào Postgres production.

---

## 2. Data Artifact Policy

Quy định quản lý dữ liệu trong dự án:

- **Source / Provenance Inputs**: Inputs nhỏ nằm trong `data/resources/`, source manifests trong `data/manifests/source/`, còn PDF/snapshot lớn nằm trong `data/heavy/raw/`.
- **Rebuildable workspace outputs**: Nằm trong `data/heavy/.work/` trong lúc build và tự xoá sau publish; failed workspace được giữ dưới `.work/failed/`.
- **Final Reproducible Contracts**: Payload nằm trong `data/heavy/processed/`, metadata snapshot nằm trong `data/manifests/corpus/`.
- **Local Experiment Outputs**: Metadata/report nhỏ nằm trong `data/retrieval_eval/`; candidates, rerank bundles, per-query reports và caches nằm trong `data/heavy/`.
