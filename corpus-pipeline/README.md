# Pipeline Xử Lý Dữ Liệu Hybrid PyMuPDF + Docling

Pipeline này chuẩn bị và tối ưu hóa dữ liệu từ **Dược thư Quốc gia Việt Nam** phục vụ cho hệ thống Retrieval-Augmented Generation (RAG) và AI Agent y dược.

---

## 1. Tổng Quan

Pipeline này chuẩn bị và tối ưu hóa dữ liệu từ **Dược thư Quốc gia Việt Nam** phục vụ cho hệ thống Retrieval-Augmented Generation (RAG) và AI Agent y dược.

---

## 2. Quick Start

Làm việc từ root của project:

```bash
cd /home/andv/personal/thesis/corpus-pipeline
```

Dự án sử dụng `uv` để quản lý môi trường và phụ thuộc:

```bash
uv sync
```

Chạy unit test suite để xác minh hệ thống:

```bash
uv run python -m unittest discover -s tests -t .
```

---

## 3. Cấu Trúc Project

- `src/corpus/`: Trích xuất text từ PDF (PyMuPDF), làm sạch markdown, phân tách section và chunking.
- `src/tables/`: Trích xuất và chuẩn hóa bảng biểu bằng Docling.
- `src/canonical/`: Xây dựng canonical blocks và tạo RAG outputs hoàn chỉnh.
- `src/rag_metadata/`: Sinh artifact metadata tinh gọn phục vụ nạp Qdrant vector database.
- `src/vector_store/`: Quản lý kết nối Qdrant và quy trình ingest vector (Local & Kaggle).
- `src/postgres_store/`: Quản lý schema trạng thái ứng dụng (Postgres app state).
- `src/evaluation/`: Sinh dataset ground truth và thực hiện đánh giá retrieval benchmark.
- `src/validation/`: Kiểm tra chất lượng và validate final RAG corpus.
- `src/cli/`: Các lệnh CLI điều khiển toàn bộ pipeline.
- `src/config/paths.py`: Khai báo các đường dẫn mặc định trong dự án.

---

## 4. Tài Liệu Hướng Dẫn Chi Tiết (Navigation Index)

Toàn bộ tài liệu chi tiết được phân tách theo từng chuyên đề trong thư mục [`docs/guides/`](docs/guides/):

| Tài liệu | Nội dung chính | Đường dẫn |
| :--- | :--- | :--- |
| **Pipeline Rebuild & Dữ Liệu** | Quy chuẩn dữ liệu vào/ra, các bước rebuild từ PDF raw tới Final RAG và kịch bản rebuild nhanh. | [`docs/guides/pipeline.md`](docs/guides/pipeline.md) |
| **Ingestion Vector & Cache** | Quản lý Qdrant collection, schema Postgres, local `llama.cpp` Docker và quy trình chạy cache vector trên Kaggle 2×T4. | [`docs/guides/vector_ingest.md`](docs/guides/vector_ingest.md) |
| **Đánh Giá Retrieval** | Quy trình sinh ground truth CSV (10,000 query) và các lệnh chạy thử nghiệm retrieval (Dense, BM25, Hybrid, Reranker). | [`docs/guides/evaluation.md`](docs/guides/evaluation.md) |
| **Vận Hành Downstream & Data Policy** | 10 quy tắc vận hành downstream agent, chiến lược context hydration và chính sách phân vùng dữ liệu. | [`docs/guides/downstream.md`](docs/guides/downstream.md) |
