# pharma-agent backend

Backend AI agent tra cứu thuốc trên corpus Dược thư Quốc gia. Thiết kế:
`docs/superpowers/specs/2026-09-11-pharma-agent-backend-design.md`.

## Chạy lần đầu

```bash
cd backend
uv sync
cp .env.example .env            # điền PHARMA_LLM__DEFAULT__API_KEY
```

Cần Qdrant (collection alias `thesis_chunks_qwen3_embedding_4b_fp16` do corpus-pipeline publish),
llama.cpp embedding (cổng 11434) và reranker (cổng 11435) từ `docker-compose.yml` ở repo root.
LLM, embedding và reranker đổi được sang OpenAI cloud hoặc server tự host chỉ bằng
`base_url`, `api_key`, `model` trong `.env`.

```bash
uv run pharma-agent check                      # kiểm tra kết nối
uv run pharma-agent ask "Paracetamol người lớn uống bao nhiêu?"
uv run pharma-agent ask "..." --json           # kèm trace đầy đủ
```

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
| `GET /api/v1/health` | Postgres, Qdrant, trạng thái agent |

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
- `skills/`: SKILL.md hệ thống (`## Tìm kiếm` cho bước judge/refine, `## Trả lời` cho bước answer).

Quy tắc phụ thuộc được kiểm tra bởi `tests/architecture/test_layering.py`.
