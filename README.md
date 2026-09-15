# Thesis: AI agent tra cứu thuốc

| Thư mục | Nội dung |
| --- | --- |
| `seed-pipeline/` | Công cụ offline: nguồn nội bộ (Dược thư Quốc gia, tờ hướng dẫn sử dụng thuốc) → knowledge bundle, embedding trên Kaggle, đánh giá retrieval |
| `backend/` | Backend AI agent (LangGraph, OpenAI SDK, Qdrant, llama.cpp) |
| `frontend/` | Web app (React Router, SSR) |
| `devops/` | Cấu hình cho image bên thứ ba mà compose dùng: nginx, CLIProxyAPI |
| `compose.yaml` | Toàn bộ stack local: Postgres, Qdrant, llama.cpp embedding và reranker, CLIProxyAPI, backend, frontend, nginx |

## Chạy toàn bộ bằng Docker

Chạy từ repo root. Các service gọi nhau bằng tên service trong mạng mặc định của Compose; trình
duyệt chỉ vào qua nginx ở `http://localhost:8080`. Port, file model và số luồng CPU đặt trong `.env`
ở root, xem `.env.example`.

Chuẩn bị một lần:

1. Tạo `backend/.env` từ `backend/.env.example`, điền `PHARMA_LLM__DEFAULT__API_KEY`, model LLM
   (ví dụ Gemini qua CLIProxyAPI trong `backend/.env.example`), `PHARMA_AUTH__JWT_SECRET` và
   `PHARMA_AUTH__CSRF_SECRET`. Compose tự đặt URL Postgres, Qdrant, LLM, embedding và reranker cho
   container, nên `backend/.env` chỉ cần key và model LLM, auth và Langfuse.
2. Tạo `devops/cli-proxy-api/config.yaml` từ `config.example.yaml` cùng thư mục và thay `api-keys`;
   `PHARMA_LLM__DEFAULT__API_KEY` phải là một trong các key đó.
3. Đăng nhập tài khoản Antigravity (Google) cho CLIProxyAPI. Lệnh in ra một URL: mở URL bằng trình
   duyệt trên cùng máy và đăng nhập; Google chuyển về `localhost:51121`, token được lưu vào volume
   `cli_proxy_auth` và lệnh tự kết thúc.

   ```bash
   docker compose run --rm -p 51121:51121 cli-proxy-api ./CLIProxyAPI --antigravity-login -no-browser
   ```

   Nhà cung cấp khác dùng cờ tương ứng (`--codex-login`, `--claude-login`, ...; xem
   `docker compose run --rm cli-proxy-api ./CLIProxyAPI --help`), cổng callback đổi được bằng
   `--oauth-callback-port`.

```bash
docker compose up -d --build
docker compose ps                              # chờ mọi service healthy
docker compose exec backend pharma-agent check
docker compose exec backend pharma-agent ask "Paracetamol người lớn uống tối đa bao nhiêu một ngày?"
```

- Hai model 4B chạy trên CPU nên lần đầu nạp model mất vài phút; backend chỉ khởi động khi
  embedding và reranker đã healthy.
- `backend-migrate` chạy `pharma-agent migrate` một lần trước khi backend khởi động.
- Cookie phiên có cờ `Secure`, trình duyệt chỉ nhận qua `http://localhost`; mở stack qua HTTP bằng
  tên máy khác (ví dụ IP LAN) thì đặt `COOKIE_SECURE=false` trong `.env` ở root.
- Đổi sang GGUF nhẹ hơn: đặt `LLAMA_EMBEDDING_MODEL` hoặc `LLAMA_RERANKER_MODEL` là tên file trong
  `ai-models/gguf`, rồi `docker compose up -d`. Embedding phải là model đã dùng khi import corpus;
  `/health` báo `CORPUS_NOT_READY` nếu metadata collection không khớp. Reranker mặc định chạy
  `native_rerank` (`/v1/rerank`), nên file GGUF phải là bản convert classifier (có tensor
  `cls.output.weight`); với GGUF thường thì đặt `LLAMA_RERANKER_PROTOCOL=completion_logprobs` và
  `LLAMA_RERANKER_RERANKING=false`.
- Dùng OpenAI cloud hoặc server OpenAI-compatible khác thay cho proxy: đặt `LLM_BASE_URL` trong
  `.env` ở root.
- Lần đầu cần nạp corpus vào container: `docker compose cp seed-pipeline/data/corpus/formulary backend:/tmp/bundle` rồi
  `docker compose exec backend pharma-agent corpus import /tmp/bundle --collection formulary --publish`.

## Môi trường Python

`backend` và `seed-pipeline` là member của một uv workspace (`pyproject.toml` ở root): chung một
`uv.lock` và một `.venv` ở root. Runtime chỉ là `backend`; `seed-pipeline` là công cụ nội bộ nên
mọi thư viện của nó nằm trong dev group, image Docker của backend không chứa chúng.

Cài một lần ở root:

```bash
uv sync
```

Chỉ chạy `uv sync` ở root: chạy trong một project sẽ gỡ khỏi `.venv` chung các gói của project
kia. Trong từng project dùng `uv run ...`, lệnh này không gỡ gói nào. Trong IDE, mở thư mục root
và chọn interpreter `.venv/bin/python`.

## Chuẩn tooling chung

Mọi project Python trong repo dùng cùng một chuẩn, không project nào được tắt rule riêng:

- **Python** 3.12, ghi trong `.python-version` ở root; môi trường do `uv` quản lý.
- **uv** pin bằng `required-version` trong `pyproject.toml` ở root; image backend và hook
  `uv-lock` dùng đúng phiên bản đó.
- **Build backend** `uv_build`.
- **Ruff**: cấu hình duy nhất ở `ruff.toml` tại root; `pyproject.toml` của từng project chỉ
  `extend` file này và khai báo `src`.
- **Pyrefly**: `pyrefly.toml` giống hệt nhau ở mọi project, không tắt nhóm lỗi nào;
  cảnh báo cũng phải bằng 0.
- **Pytest**: `asyncio_mode = "auto"`, `--import-mode=importlib`, test cần Docker gắn marker
  `integration`.
- **Phiên bản dev tool** (`ruff`, `pyrefly`, `pytest`, `pytest-asyncio`, `pre-commit`) pin một lần
  trong dev group của `pyproject.toml` ở root.
- **Dockerfile** nằm cạnh code nó build, kèm `Dockerfile.dockerignore`; hadolint kiểm tra trong
  pre-commit.

Lỗi lint hoặc lỗi kiểu được sửa trong code, không dùng `# noqa`, `# type: ignore` hay tắt rule.

## Kiểm tra

Chạy trong từng project:

```bash
uv run ruff check src tests && uv run ruff format --check src tests
uv run pyrefly check --min-severity warn
uv run pytest -q
```

Pre-commit dùng chung cho cả repo, cài một lần từ root:

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```
