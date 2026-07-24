# Hướng Dẫn Rebuild Pipeline Và Cấu Trúc Dữ Liệu

Tài liệu này chi tiết quy trình xử lý dữ liệu từ file PDF gốc **Dược thư Quốc gia Việt Nam** thành corpus RAG chuẩn hóa (sections và chunks).

---

## 1. Dữ Liệu Vào / Ra

### 1.1 Input Chính

- `data/raw/duoc-thu-quoc-gia-viet-nam.pdf`: PDF gốc của Dược thư Quốc gia Việt Nam.
- `data/raw/curation/docling_tables.jsonl`: Bảng Docling đã được curate, dùng khi build canonical/final RAG.
- `data/raw/vietnamese_valid_syllables.json`: Danh sách âm tiết tiếng Việt hợp lệ, dùng để validate các rule sửa lỗi tách âm tiết trong bước clean text/table.

### 1.2 Output Trung Gian

- Các text/section/chunk trung gian được ghi trong `data/.work/<build-id>/` và xoá sau build thành công.

### 1.3 Output Cuối

Output cuối cho downstream agent nằm trong `data/processed/rag-final/`:

- `sections.jsonl`: Đơn vị kiến thức logic, có `section_id`, `table_ids`, `source_mix`, page range, context, `hydrate_strategy`, và `section_char_count`. Section có bảng chứa Markdown table đã được ghép vào cuối text. Đây là artifact offline để kiểm toán và build chunk payload.
- `chunks.jsonl`: Đơn vị dùng để embedding/retrieval, có `section_id` để hydrate ngược về section, đồng thời mang `hydrate_strategy`, `section_char_count`, và `chunk_role` của section cha. Chunk được sinh từ canonical blocks; paragraph block dùng splitter văn bản thường, table block dùng table-aware splitter để không cắt mất header cột, còn index/list dài dùng splitter theo entry/dòng.
- `chunks.jsonl`: Lean Qdrant-ready chunk artifact cho ingest vector.
- `manifest.json` và `audit.json`: Metadata và báo cáo chất lượng.

### 1.4 Hydrate Strategy Overview

`hydrate_strategy` hỗ trợ 3 giá trị:

- `full_section`: Section không quá dài (`<= 16 000` ký tự), downstream có thể lấy nguyên section context.
- `chunk_window`: Section dài (`> 16 000` ký tự), downstream nên lấy chunk hit và một số chunk lân cận thay vì nhét nguyên section vào prompt.
- `search_only`: Section quá lớn/dạng index (hiện dùng cho `Bảng tra cứu biệt dược`), chỉ dùng chunk hit hoặc xử lý bằng index lookup.

---

## 5. Pipeline Rebuild Chính

### 5.1 Trích Xuất Text Bằng PyMuPDF

```bash
uv run python -m cli.extract_pymupdf_text
```

Lệnh này đọc PDF gốc và ghi output được chỉ định (thường dưới `data/.work/<build-id>/text/`) với page marker dạng:

```text
<!-- page 0758 -->
```

### 5.2 Làm Sạch Markdown

```bash
uv run python -m cli.clean_markdown_corpus
```

Cleaner chỉ normalize format, heading, khoảng trắng, ký hiệu, và một số lỗi split-word có kiểm soát. Không dùng broad OCR replacement có thể làm hỏng từ đúng như `nồng độ`.

### 5.3 Tạo Section Và Chunk Text

```bash
uv run python -m cli.preprocess_rag_corpus
```

Parser đọc `full.cleaned.md`, tạo section theo chuyên luận chung/chuyên luận thuốc, rồi chia chunk để retrieval. Mặc định chunk upper bound là `3000` ký tự; dùng `--max-chars` chỉ khi cần chạy thí nghiệm khác.

Rule bảo vệ coverage:
- Giữ đủ 20 chuyên luận chung chính thức.
- Giữ đủ vùng chuyên luận thuốc từ page 99-1496.
- Giữ phụ lục 1, phụ lục 2, phụ lục 3/ATC và mục tra cứu biệt dược.
- Chỉ bỏ qua các page tiêu đề/blank đã biết như 37-38 và 1497-1498.

### 5.4 Trích Xuất Bảng Docling Khi Cần

Nếu cần phát hiện bảng lại từ PDF gốc:

```bash
uv run python -m cli.extract_docling_tables \
  --ocr off \
  --batch-size 50 \
  --out data/.work/manual/docling
```

Docling không ghi đè text PyMuPDF. OCR mặc định nên để off cho PDF này.

### 5.5 Curate Bảng Docling

```bash
uv run python -m cli.curate_docling_tables \
  --input data/.work/manual/docling \
  --sections data/.work/manual/rag/sections.jsonl
```

Lệnh này chuẩn hóa bảng, gộp bảng cắt qua trang, sửa các artifact Docling/OCR phổ biến trong Markdown table, và map bảng vào section phù hợp.

### 5.6 Build Canonical Và Final RAG

```bash
uv run python -m cli.build_canonical_rag
```

Canonical output kết hợp paragraph text từ PyMuPDF với Markdown table từ Docling, rồi ghi final RAG vào `data/processed/rag-final/`. Bảng được lưu thành canonical block riêng (`docling_table`) và được ghép vào text của final section để hydrate đầy đủ sau retrieval.

Final chunks được sinh từ canonical blocks thay vì cắt nguyên section đã ghép bảng:
- Paragraph block được cắt bằng text splitter thường.
- Table block được cắt theo nhóm row; mỗi table chunk lặp lại header và separator Markdown.
- Brand index được cắt theo từng entry hoàn chỉnh, không cắt giữa dòng `- **Biệt dược**: ...`.
- Các appendix/list dài đã biết được cắt theo dòng để giảm khả năng cắt giữa mục tra cứu.
- Table chunk có thêm `chunk_content_type="table"`, `table_id`, `table_chunk_index`, `table_chunk_count`, và `source_block_id`.
- Paragraph chunk có `chunk_content_type="paragraph"` và `source_block_id`.

`chunk_role` mô tả vai trò retrieval của chunk:
- `prose`: Chunk văn bản thường.
- `table`: Chunk bảng có lặp header/separator Markdown.
- `index_entry`: Chunk mục tra cứu biệt dược, gom các entry hoàn chỉnh.
- `appendix_list`: Chunk phụ lục/danh sách dài, gom theo dòng hoàn chỉnh.

Build canonical có allowlist nhỏ để merge 14 cặp `part-*` ngắn đã audit là continuation chắc chắn. Ngoài ra có lớp correction hẹp cho các fragment đã đối chiếu thủ công với PDF gốc.

### 5.7 Validate Final RAG

```bash
uv run python -m cli.validate_final_rag
```

Validator là quality gate cuối trước khi ingest. Lệnh này kiểm tra JSONL, duplicate id, reference integrity, section/chunk không mồ côi, warning, giới hạn `max-chars`, cắt từ ở chunk boundary, chunk bảng thiếu Markdown separator/header, chunk brand-index bị cắt giữa entry, coverage page nội dung chính, đủ 20 chuyên luận chung chính thức, sentinel BSA/Vancomycin, Unicode combining mark, dính số-đơn vị, punctuation dính, và các text-quality/OCR artifacts đã biết.

Mặc định report được ghi vào `data/processed/rag-final/final_validation_report.json` và `data/processed/rag-final/final_validation_report.md`. Đổi vị trí report bằng `--output-json` và `--output-md`.

---

## 6. Rebuild Nhanh Theo Tình Huống

### 3.1 Rebuild Toàn Bộ

```bash
uv sync

uv run python -m cli.extract_pymupdf_text
uv run python -m cli.clean_markdown_corpus
uv run python -m cli.preprocess_rag_corpus

uv run python -m cli.curate_docling_tables \
  --input data/.work/manual/docling \
  --sections data/.work/manual/rag/sections.jsonl

uv run python -m cli.build_canonical_rag
uv run python -m cli.validate_final_rag
uv run python -m unittest discover -s tests -t .
```

### 3.2 Rebuild Từ Canonical / Final Logic

Nếu chỉ thay đổi logic canonical/final hoặc metadata hydrate, chạy lại từ bước build canonical:

```bash
uv run python -m cli.build_canonical_rag
uv run python -m cli.validate_final_rag
uv run python -m unittest discover -s tests -t .
```
