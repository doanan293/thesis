# pharma-agent backend

Backend AI agent tra cứu thuốc trên corpus Dược thư Quốc gia. Thiết kế:
`docs/superpowers/specs/2026-09-11-pharma-agent-backend-design.md`.

## Chạy lần đầu

```bash
cd backend
uv sync
cp .env.example .env            # điền PHARMA_LLM__DEFAULT__API_KEY
```

Cần Postgres, Qdrant, llama.cpp embedding (cổng 11434) và reranker (cổng 11435) từ
`docker-compose.yml` ở repo root. Corpus nằm trong schema `corpus` của Postgres; Qdrant chỉ là
index (alias `chunks_current`). Nạp một knowledge bundle và publish trước khi hỏi:

```bash
uv run pharma-agent migrate
uv run pharma-agent corpus import <bundle_dir> --collection formulary --publish
```

LLM, embedding và reranker đổi được sang OpenAI cloud hoặc server tự host chỉ bằng
`base_url`, `api_key`, `model` trong `.env`.

```bash
uv run pharma-agent check                      # kiểm tra kết nối
uv run pharma-agent ask "Paracetamol người lớn uống bao nhiêu?"
uv run pharma-agent ask "..." --json           # kèm trace đầy đủ
```

## Chạy toàn bộ bằng Docker

Chạy từ repo root. Cần `backend/.env` có LLM key và `PHARMA_AUTH__JWT_SECRET`. Port, file model
và số luồng CPU đặt trong `.env` ở root, xem `.env.example`.

```bash
docker compose up -d --build
docker compose ps                              # chờ mọi service healthy
docker compose exec backend pharma-agent check
docker compose exec backend pharma-agent ask "Paracetamol người lớn uống tối đa bao nhiêu một ngày?"
```

- Hai model 4B chạy trên CPU nên lần đầu nạp model mất vài phút; backend chỉ khởi động khi
  embedding và reranker đã healthy.
- `backend-migrate` chạy `pharma-agent migrate` một lần trước khi backend khởi động.
- Backend dùng mạng host để gọi được proxy LLM đang lắng nghe `127.0.0.1` trên Windows
  (WSL cần `networkingMode=mirrored`). Compose tự đặt URL Postgres, Qdrant, embedding và
  reranker theo port ở `.env` root, nên `backend/.env` chỉ cần LLM, auth và Langfuse.
- Đổi sang GGUF nhẹ hơn: đặt `LLAMA_EMBEDDING_MODEL` hoặc `LLAMA_RERANKER_MODEL` là tên file trong
  `ai-models/gguf`, rồi `docker compose up -d`. Embedding phải là model đã dùng khi
  import corpus; `/health` báo `CORPUS_NOT_READY` nếu metadata collection không khớp.
- Lần đầu cần nạp corpus vào container: `docker compose cp <bundle_dir> backend:/tmp/bundle` rồi
  `docker compose exec backend pharma-agent corpus import /tmp/bundle --collection formulary --publish`.

## Chạy HTTP API

```bash
docker compose up -d postgres qdrant llama-embedding llama-reranker   # chạy từ repo root
cd backend
uv run pharma-agent migrate
uv run pharma-agent serve            # http://127.0.0.1:8000/docs
```

Cần thêm `PHARMA_AUTH__JWT_SECRET` (ít nhất 32 ký tự) trong `.env`. Không có
`PHARMA_LLM__DEFAULT__API_KEY` thì API vẫn chạy, riêng `/chat` trả 503.

Thử nhanh bằng curl:

```bash
curl -s -X POST localhost:8000/api/v1/auth/register -H 'content-type: application/json' \
  -d '{"email":"an@example.com","password":"S3cure-password!"}'
TOKEN=$(curl -s -X POST localhost:8000/api/v1/auth/jwt/login \
  -d 'username=an@example.com&password=S3cure-password!' \
  | python -c 'import sys, json; print(json.load(sys.stdin)["access_token"])')
curl -N -X POST localhost:8000/api/v1/chat/stream -H "Authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' -d '{"message":"Paracetamol người lớn uống bao nhiêu?"}'
```

Luồng SSE lần lượt gồm `conversation`, `phase`, `skills_selected`, `evidence`, `token`,
`citations`, `done`. Nếu không lưu được lượt hội thoại thì có thêm `error` với mã `PERSIST_FAILED`.

| Endpoint | Mô tả |
| --- | --- |
| `POST /api/v1/auth/register`, `/auth/jwt/login`, `/auth/google/*` | Đăng ký, đăng nhập (JWT), Google OAuth khi đã cấu hình |
| `GET /api/v1/users/me` | Người dùng hiện tại |
| `POST /api/v1/chat`, `/chat/stream` | Hỏi đáp một lượt (JSON hoặc SSE) |
| `GET/PATCH/DELETE /api/v1/conversations/{id}`, `GET .../messages` | Lịch sử hội thoại |
| `GET /api/v1/health` | Postgres, `corpus` (collection Qdrant khớp model embedding và mọi collection trong `PHARMA_RETRIEVAL__COLLECTIONS` có release hiện hành; nếu không, `reasons.corpus = "CORPUS_NOT_READY"`), trạng thái agent |
| `GET/POST /api/v1/skills`, `PATCH/DELETE /api/v1/skills/{id}` | Skill hệ thống và skill tự tải lên (`SKILL.md` tối đa 64 KB) |
| `POST /api/v1/messages/{id}/feedback` | Đánh giá câu trả lời (`up`/`down`), gửi thêm score sang Langfuse |

## Quan sát và dọn dẹp

Đặt `PHARMA_LANGFUSE__PUBLIC_KEY`, `PHARMA_LANGFUSE__SECRET_KEY` và `PHARMA_LANGFUSE__HOST`
để bật Langfuse. Mỗi lượt hỏi đáp là một trace tên `chat_turn`, gắn `session` là hội thoại
và `user` là người dùng. Các bước của graph, mọi lời gọi LLM và embedding, cùng bước rerank đều nằm trong trace đó. Feedback
của người dùng được ghi thành score `user_feedback` (1 là up, 0 là down) trên cùng trace.

Skill hệ thống trong `skills/` được đồng bộ vào Postgres mỗi lần service khởi động.

Checkpoint LangGraph cũ hơn `PHARMA_CHECKPOINTS__RETENTION_DAYS` ngày (mặc định 7) được xóa
một lần khi service khởi động. Có thể chạy tay:

```bash
uv run pharma-agent cleanup-checkpoints --days 7
```

## Kiểm tra

```bash
uv run ruff check src tests && uv run ruff format --check src tests
uv run pyrefly check
uv run pytest -q                 # unit
uv run pytest -q -m integration  # cần Docker (Qdrant thật)
```

## Bố cục

- `src/pharma_agent/domain`: Python thuần, không framework. Aggregate `AgentRun`, retrieval,
  skill, guardrail, prompt.
- `src/pharma_agent/application`: vòng lặp agentic RAG trên LangGraph và progress events.
- `src/pharma_agent/infrastructure`: OpenAI, Qdrant, llama.cpp, settings, composition root.
- `skills/`: skill hệ thống theo [Agent Skills specification](https://agentskills.io/specification): `skills/<name>/SKILL.md`, `name` trùng tên thư mục, `name` chỉ gồm chữ thường a-z, số và gạch nối, body là hướng dẫn tự do. Kiểm tra bằng `uv run agentskills validate skills/<name>` (CLI của thư viện tham chiếu `skills-ref`).

Quy tắc phụ thuộc được kiểm tra bởi `tests/architecture/test_layering.py`.
