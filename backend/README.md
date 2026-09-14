# pharma-agent backend

Backend AI agent tra cứu thuốc trên corpus Dược thư Quốc gia. Thiết kế:
`docs/superpowers/specs/2026-09-11-pharma-agent-backend-design.md`.

## Chạy lần đầu

```bash
uv sync                         # ở repo root: một .venv chung cho cả uv workspace
cd backend
cp .env.example .env            # điền PHARMA_LLM__DEFAULT__API_KEY
```

Cần Postgres, Qdrant, llama.cpp embedding (cổng 11434), reranker (cổng 11435) và proxy LLM
CLIProxyAPI (cổng 8317) từ `compose.yaml` ở repo root. Corpus nằm trong schema `corpus` của Postgres; Qdrant chỉ là
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

Stack đầy đủ (backend, frontend, nginx và các service phụ trợ) chạy bằng `compose.yaml` ở repo
root; xem mục *Chạy toàn bộ bằng Docker* trong README ở root.

## Chạy HTTP API

```bash
docker compose up -d postgres qdrant llama-embedding llama-reranker cli-proxy-api   # chạy từ repo root
cd backend
uv run pharma-agent migrate
uv run pharma-agent serve            # http://127.0.0.1:8000/docs
```

Cần `PHARMA_AUTH__JWT_SECRET` và `PHARMA_AUTH__CSRF_SECRET` (mỗi giá trị ít nhất 32 ký tự) trong
`.env`. Không có `PHARMA_LLM__DEFAULT__API_KEY` thì API vẫn chạy, riêng chat trả 503
`AGENT_UNAVAILABLE`.

### Xác thực

- **Trình duyệt**: cookie `pharma_session` (HttpOnly, `SameSite=Lax`, `Secure` theo
  `PHARMA_AUTH__COOKIE_SECURE`, sống `PHARMA_AUTH__SESSION_LIFETIME_SECONDS` giây, mặc định 7 ngày).
  Phiên lưu trong bảng `access_tokens`; đăng xuất xoá phiên trong DB.
- **CSRF**: request có cookie `pharma_session` với method không an toàn (POST, PATCH, PUT, DELETE)
  phải gửi header `x-csrftoken` bằng giá trị cookie `csrftoken` (cookie này có sau request GET
  đầu tiên, ví dụ `/users/me`). Thiếu hoặc sai trả 403 `CSRF_FAILED`.
- **CLI và test**: bearer JWT qua `/auth/jwt/login`, không bị kiểm tra CSRF.
- **Google OAuth** (khi đặt `PHARMA_AUTH__GOOGLE_CLIENT_ID` và `PHARMA_AUTH__GOOGLE_CLIENT_SECRET`):
  frontend gọi `GET /api/v1/auth/google/authorize`, Google chuyển về
  `{PHARMA_AUTH__FRONTEND_URL}/auth/google/callback`, frontend gọi
  `GET /api/v1/auth/google/callback?code=…&state=…` và nhận 204 kèm cookie `pharma_session`.

Luồng cookie bằng curl (curl coi `localhost` là origin an toàn nên giữ cookie `Secure`):

```bash
JAR=$(mktemp)
curl -s -c "$JAR" -b "$JAR" -X POST localhost:8000/api/v1/auth/register \
  -H 'content-type: application/json' -d '{"email":"an@example.com","password":"S3cure-password!"}'
curl -s -c "$JAR" -b "$JAR" -X POST localhost:8000/api/v1/auth/cookie/login \
  -d 'username=an@example.com&password=S3cure-password!'
curl -s -c "$JAR" -b "$JAR" localhost:8000/api/v1/users/me          # nhận thêm cookie csrftoken
CSRF=$(awk '$6 == "csrftoken" {print $7}' "$JAR")
CONVERSATION=$(curl -s -c "$JAR" -b "$JAR" -X POST localhost:8000/api/v1/conversations \
  -H "x-csrftoken: $CSRF" | python -c 'import sys, json; print(json.load(sys.stdin)["id"])')
curl -N -c "$JAR" -b "$JAR" -X POST localhost:8000/api/v1/chat/stream -H "x-csrftoken: $CSRF" \
  -H 'content-type: application/json' \
  -d "{\"message\":\"Paracetamol người lớn uống bao nhiêu?\",\"conversation_id\":\"$CONVERSATION\"}"
curl -s -c "$JAR" -b "$JAR" -X POST localhost:8000/api/v1/auth/cookie/logout -H "x-csrftoken: $CSRF"
```

Luồng bearer:

```bash
TOKEN=$(curl -s -X POST localhost:8000/api/v1/auth/jwt/login \
  -d 'username=an@example.com&password=S3cure-password!' \
  | python -c 'import sys, json; print(json.load(sys.stdin)["access_token"])')
CONVERSATION=$(curl -s -X POST localhost:8000/api/v1/conversations -H "Authorization: Bearer $TOKEN" \
  | python -c 'import sys, json; print(json.load(sys.stdin)["id"])')
curl -N -X POST localhost:8000/api/v1/chat/stream -H "Authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' \
  -d "{\"message\":\"Paracetamol người lớn uống bao nhiêu?\",\"conversation_id\":\"$CONVERSATION\"}"
```

### Dạng stream

`POST /api/v1/chat/stream` theo AI SDK UI Message Stream v1 (encoder `pharma_agent.api.ui_stream`).
Header phản hồi: `x-vercel-ai-ui-message-stream: v1`, `Cache-Control: no-cache`,
`X-Accel-Buffering: no`. Mỗi frame chỉ có một dòng `data: {json}` (không có `event:`), ping
keepalive dạng comment mỗi 15 giây, frame cuối là `data: [DONE]`.

| Chunk | Nội dung |
| --- | --- |
| `{"type":"start","messageId":…}` | Chunk đầu; `messageId` là ID tin nhắn trợ lý |
| `{"type":"data-conversation","transient":true,"data":{"id":…,"title":…}}` | Chunk thứ hai; không lưu vào tin nhắn |
| `{"type":"data-phase","id":"phase","data":{"phase":…,"round":…}}` | `phase` là `guarding`, `understanding`, `selecting_skills`, `searching`, `reading` hoặc `answering`; `round` chỉ có khi `searching`; cùng `id` nên client cập nhật tại chỗ |
| `{"type":"data-skills","data":{"skills":[{"name":…,"title":…}]}}` | Skill đã chọn |
| `{"type":"data-evidence","id":"evidence","data":{"items":[…]}}` | Mỗi item: `index`, `source`, `title`, `section`, `startPage`, `endPage` (`null` khi không có), `snippet` |
| `{"type":"text-start","id":"text"}`, `{"type":"text-delta","id":"text","delta":…}`, `{"type":"text-end","id":"text"}` | Câu trả lời; marker `[n]` không bị cắt giữa hai `text-delta` |
| `{"type":"source-document","sourceId":…,"mediaType":"text/markdown","title":…,"providerMetadata":{"pharma":{…}}}` | Sau `text-end`, một chunk cho mỗi `[n]`: `sourceId` là `chunk_version_id`, `title` là `"{title} › {section}"`, `pharma` gồm `index`, `source`, `title`, `section`, `startPage`, `endPage`, `snippet`, `isCurrent` |
| `{"type":"finish","finishReason":…,"messageMetadata":{…}}` | Chunk cuối trước `[DONE]`; `finishReason` là `error` khi `status` là `error` hoặc `timeout`, còn lại `stop`; `messageMetadata` gồm `status`, `errorCode`, `usage`, `runId`, `persisted`, `createdAt` |

Mọi lỗi REST và lỗi trước khi stream bắt đầu trả `application/problem+json` (RFC 9457):
`type`, `title`, `status`, `detail`, `code`, thêm `errors` khi 422.

| Endpoint (tiền tố `/api/v1`) | Mô tả |
| --- | --- |
| `POST /auth/register` | Đăng ký |
| `POST /auth/cookie/login`, `POST /auth/cookie/logout` | Đăng nhập, đăng xuất bằng cookie (trình duyệt) |
| `POST /auth/jwt/login`, `POST /auth/jwt/logout` | Bearer JWT (CLI, test) |
| `GET /auth/google/authorize`, `GET /auth/google/callback` | Google OAuth, đăng nhập bằng cookie |
| `GET/PATCH /users/me`, `GET/PATCH/DELETE /users/{id}` | Người dùng hiện tại; quản trị người dùng (superuser) |
| `POST /chat/stream`, `POST /chat` | Hỏi đáp một lượt (UI Message Stream hoặc JSON) |
| `POST /conversations`, `GET /conversations?limit=&cursor=` | Tạo hội thoại; danh sách `{items, next_cursor}` |
| `GET/PATCH/DELETE /conversations/{id}` | Xem, đổi tên, xoá hội thoại |
| `GET /conversations/{id}/messages?limit=&cursor=` | Lịch sử dạng `UIMessage`, phân trang cursor |
| `GET /messages/{message_id}/citations/{index}` | Toàn văn khối mà mô hình đã đọc cho citation |
| `POST /messages/{message_id}/feedback` | Đánh giá câu trả lời (`up`/`down`), gửi score sang Langfuse |
| `GET/POST /skills`, `PATCH/DELETE /skills/{id}` | Skill hệ thống và skill tự tải lên (`SKILL.md` tối đa 64 KB) |
| `GET /health` | Postgres, `corpus` (collection Qdrant khớp model embedding và mọi collection trong `PHARMA_RETRIEVAL__COLLECTIONS` có release hiện hành; nếu không, `reasons.corpus = "CORPUS_NOT_READY"`), trạng thái agent |

Schema OpenAPI cho frontend: `uv run pharma-agent export-openapi --output ../frontend/openapi.json`.

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

Phiên cookie hết hạn không tự bị xoá khỏi `access_tokens` (fastapi-users chỉ bỏ qua chúng).
Chạy định kỳ:

```bash
uv run pharma-agent cleanup-sessions
```

## Server E2E

Playwright của frontend chạy backend bằng server E2E: app thật trên Postgres và Qdrant thật,
LLM giả theo kịch bản và embedding giả 4 chiều. Mọi thứ giả nằm trong `tests/`, code production
không có cờ "chế độ giả".

```bash
docker compose up -d postgres qdrant     # chạy từ repo root
cd backend
uv run python -m tests.e2e.server --port 8001
```

- Mỗi lần khởi động: xoá và tạo lại database của `E2E_POSTGRES_DSN` (mặc định
  `postgresql+psycopg://thesis:thesis@localhost:5433/pharma_e2e`; tên database phải chứa `e2e`),
  chạy migration, xoá collection `chunks_fake_embedding_4d` trong `E2E_QDRANT_URL` (mặc định
  `http://localhost:6333`), import và publish bundle `tests/fixtures/knowledge_bundle_small`.
  Index dùng alias riêng `e2e_chunks_current` (`PHARMA_RETRIEVAL__QDRANT_COLLECTION`), nên database dev
  `thesis` và alias `chunks_current` không bị động tới.
- Kịch bản theo câu hỏi: chứa `[e2e:blocked]` thì lượt kết thúc `blocked`; chứa `[e2e:timeout]`
  thì `timeout` sau 5 giây; còn lại là câu trả lời có citation `[1]`.
- Cookie phiên không đặt `Secure` để mọi trình duyệt nhận qua `http://`.

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
