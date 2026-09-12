# Thesis: AI agent tra cứu thuốc

| Thư mục | Nội dung |
| --- | --- |
| `corpus-pipeline/` | Xây corpus Dược thư Quốc gia + An Khang, embedding, Qdrant, đánh giá retrieval |
| `backend/` | Backend AI agent (LangGraph, OpenAI SDK, Qdrant, llama.cpp) |
| `docker-compose.yml` | Toàn bộ stack local: Qdrant, Postgres, llama.cpp embedding và reranker, backend |

## Chuẩn tooling chung

Mọi project Python trong repo dùng cùng một chuẩn, không project nào được tắt rule riêng:

- **Python** 3.12, ghi trong `.python-version` của từng project; môi trường do `uv` quản lý.
- **Build backend** `uv_build`.
- **Ruff**: cấu hình duy nhất ở `ruff.toml` tại root; `pyproject.toml` của từng project chỉ
  `extend` file này và khai báo `src`.
- **Pyrefly**: `pyrefly.toml` giống hệt nhau ở mọi project, không tắt nhóm lỗi nào;
  cảnh báo cũng phải bằng 0.
- **Pytest**: `asyncio_mode = "auto"`, `--import-mode=importlib`, test cần Docker gắn marker
  `integration`.
- **Phiên bản dev tool** được pin giống nhau (`ruff`, `pyrefly`, `pytest`, `pytest-asyncio`,
  `pre-commit`).

Lỗi lint hoặc lỗi kiểu được sửa trong code, không dùng `# noqa`, `# type: ignore` hay tắt rule.

## Kiểm tra

Chạy trong từng project:

```bash
uv sync
uv run ruff check src tests && uv run ruff format --check src tests
uv run pyrefly check --min-severity warn
uv run pytest -q
```

Pre-commit dùng chung cho cả repo, cài một lần từ root:

```bash
uv run --project backend pre-commit install
uv run --project backend pre-commit run --all-files
```
