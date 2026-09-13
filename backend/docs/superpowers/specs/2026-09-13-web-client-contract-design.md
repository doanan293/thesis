# Thiết kế contract backend cho web client

Ngày: 2026-09-13. Trạng thái: đã duyệt qua brainstorming, chờ implementation plan.

Spec liên quan, cùng đợt:

- `backend/docs/superpowers/specs/2026-09-13-corpus-platform-design.md` (corpus platform, gọi tắt là spec C). Spec này phụ thuộc schema `corpus` của spec C.
- `frontend/docs/superpowers/specs/2026-09-13-frontend-v1-design.md` (frontend v1, gọi tắt là spec B).

Spec này thay thế các phần sau của `2026-09-11-pharma-agent-backend-design.md`: protocol SSE tự định nghĩa (§8.5), auth JWT bearer cho trình duyệt (§8.4), dạng lỗi và cách phân trang của API.

## 1. Mục tiêu

Backend phục vụ frontend React Router (spec B) theo các chuẩn được cộng đồng dùng rộng rãi:

1. Chat stream theo **AI SDK UI Message Stream v1**, thay protocol SSE tự định nghĩa.
2. History trả về dạng **`UIMessage`** của AI SDK, phân trang bằng **cursor keyset**.
3. Auth trình duyệt bằng **cookie HttpOnly cùng origin**, session lưu Postgres, có **chống CSRF**; bearer JWT giữ cho CLI và test.
4. Lỗi theo **RFC 9457 Problem Details**.
5. **OpenAPI** sạch để frontend sinh client bằng orval.
6. Citation xem được **toàn văn** khối mà mô hình đã đọc.
7. Có **server E2E** để Playwright chạy toàn bộ luồng với LLM giả.

## 2. Quyết định đã chốt

| Chủ đề | Quyết định | Lựa chọn khác đã cân nhắc |
| --- | --- | --- |
| Protocol stream | AI SDK UI Message Stream v1 (header `x-vercel-ai-ui-message-stream: v1`) | AG-UI (spec còn draft, client 0.0.x, ít người dùng hơn); giữ SSE tự định nghĩa và viết adapter ở frontend |
| Vị trí mapping | Encoder mỏng ở tầng API; domain `ProgressEvent` giữ nguyên | Đổi domain sang khái niệm của AI SDK |
| History | `UIMessage` dùng chung model với encoder | Frontend tự chuyển đổi |
| Tạo hội thoại | `POST /conversations` trước tin nhắn đầu | Chỉ tạo ngầm khi mở stream |
| Phân trang | Cursor keyset mã hoá, response `{items, next_cursor}` | Cursor datetime hiện tại (trùng hoặc sót khi cùng thời điểm) |
| Citation | Tham chiếu chunk version bất biến (spec C); xem trước ở mức khối, toàn văn qua endpoint riêng | Lưu bản sao text; chọn câu khớp |
| Auth trình duyệt | fastapi-users `CookieTransport` + `DatabaseStrategy`, cùng origin qua nginx | BFF trong React Router; JWT trong `localStorage` |
| CSRF | `starlette-csrf` với `sensitive_cookies` | Chỉ dựa vào `SameSite=Lax` |
| Lỗi | RFC 9457 qua exception handler của FastAPI | Giữ ba dạng lỗi hiện có |

## 3. Chat stream

### 3.1 Endpoint

`POST /api/v1/chat/stream`, body `ChatRequest {message: str (1–4000), conversation_id: str | null}` (giữ nguyên). Frontend luôn gửi `conversation_id` đã tạo qua `POST /conversations` và chỉ gửi tin nhắn cuối (dùng `prepareSendMessagesRequest` của AI SDK).

Response `text/event-stream`:

- Mỗi chunk là một dòng `data: {json}` rồi dòng trống; kết thúc bằng `data: [DONE]`.
- Header `x-vercel-ai-ui-message-stream: v1`, `Cache-Control: no-cache`, `X-Accel-Buffering: no`.
- Ping keepalive dạng comment mỗi 15 giây (giữ `sse-starlette`).
- Lỗi trước khi stream bắt đầu (401, 404, 422, 503) trả `application/problem+json` (§7).
- Client ngắt kết nối thì graph bị huỷ như hiện nay; lượt đó không được lưu.

### 3.2 Encoder `api/ui_stream.py`

Hàm thuần chuyển `ProgressEvent` thành chunk. Model Pydantic của chunk và của `UIMessage` dùng tên field camelCase theo AI SDK (alias generator); các model REST khác giữ snake_case.

| Domain event | Chunk AI SDK |
| --- | --- |
| Mở turn | `{"type":"start","messageId":"<assistant_message_id>"}` |
| `conversation` | `{"type":"data-conversation","transient":true,"data":{"id","title"}}` |
| `phase` | `{"type":"data-phase","id":"phase","data":{"phase","round"}}`; cùng `id` nên client cập nhật tại chỗ |
| `skills_selected` | `{"type":"data-skills","data":{"skills":[{"name","title"}]}}` |
| `evidence` | `{"type":"data-evidence","id":"evidence","data":{"items":[EvidenceItem]}}` |
| `token` đầu tiên | `{"type":"text-start","id":"text"}` rồi `{"type":"text-delta","id":"text","delta"}` |
| `token` tiếp theo | `text-delta` |
| `citations` | `{"type":"text-end","id":"text"}` rồi mỗi citation một `source-document` (§3.4) |
| `done` | `{"type":"finish","finishReason","messageMetadata":{"status","errorCode","usage","runId","persisted","createdAt"}}` rồi `data: [DONE]` |

- `phase ∈ {guarding, understanding, selecting_skills, searching, reading, answering}`; `round` chỉ có với `searching`.
- `status ∈ {completed, partial, abstained, blocked, redirected, error, timeout}`; `finishReason = "error"` khi `status ∈ {error, timeout}`, còn lại `"stop"`.
- `EvidenceItem = {index, source, title, section, startPage, endPage, snippet}`; trang là `null` khi không có.

### 3.3 Thay đổi domain và application

- `ChatService.open_turn` tạo sẵn `assistant_message_id` và `user_message_id`; `build_turn_messages` dùng lại hai ID này.
- Event `done` mang `persisted: bool`; bỏ event `error` `PERSIST_FAILED` gửi sau `done`. CLI đọc cờ `persisted` để in cảnh báo.
- Hàm `make_snippet(text, max_chars)` trong domain: gộp khoảng trắng, bỏ ký hiệu bảng markdown, cắt theo ranh giới từ, thêm "…". Dùng cho `EvidenceItem.snippet` (200 ký tự), `CitationView.snippet` (200 ký tự) và `EvidenceSet.summary_view` (300 ký tự, thay đoạn cắt đang viết riêng).

### 3.4 `source-document` cho citation

```json
{"type":"source-document","sourceId":"<chunk_version_id>","mediaType":"text/markdown",
 "title":"Paracetamol › Liều lượng và cách dùng",
 "providerMetadata":{"pharma":{"index":1,"source":"Dược thư Quốc gia Việt Nam (Xuất bản lần thứ 2) – …",
   "title":"Paracetamol","section":"Liều lượng và cách dùng",
   "startPage":812,"endPage":813,"snippet":"Người lớn và trẻ em trên 12 tuổi…","isCurrent":true}}}
```

- `snippet` là khoảng 200 ký tự đầu của `chunk_text` của chunk khớp. Frontend ghi "Trích từ mục {title} › {section}".
- `isCurrent` luôn là `true` trong stream; trong history có thể là `false` (§4.2).
- Text giữ marker `[n]` đã được `CitationSanitizer` làm sạch.

### 3.5 Bảo đảm của stream (có test)

1. Mỗi `[n]` trong text có đúng một `source-document` với `index = n`.
2. Mỗi `source-document` được nhắc ít nhất một lần trong text.
3. Marker không bị cắt ngang giữa hai `text-delta`.
4. Mọi chunk hợp lệ theo `uiMessageChunkSchema` của AI SDK (kiểm ở contract test phía frontend, §10).

## 4. Hội thoại và tin nhắn

### 4.1 Endpoint

| Method, path | Thay đổi |
| --- | --- |
| `POST /conversations` | **Mới.** Tạo hội thoại tiêu đề mặc định "Cuộc trò chuyện mới", trả 201 `ConversationView`. Turn đầu tiên đặt tiêu đề từ tin nhắn đầu nếu tiêu đề vẫn là mặc định |
| `GET /conversations?limit=20&cursor=` | Trả `{items: ConversationView[], next_cursor: str \| null}`; sắp xếp `(updated_at desc, id desc)`; **bỏ qua hội thoại có `turn_count = 0`** |
| `GET /conversations/{id}` | Giữ nguyên |
| `PATCH /conversations/{id}`, `DELETE /conversations/{id}` | Giữ nguyên |
| `GET /conversations/{id}/messages?limit=30&cursor=` | Trả `{items: UIMessage[], next_cursor}`. Trang đầu (không cursor) là 30 tin nhắn mới nhất; `items` trong một trang sắp xếp cũ → mới; `next_cursor` trỏ tới trang cũ hơn |

- Cursor là base64url của JSON `{"t": "<timestamp ISO>", "id": "<uuid>"}`, keyset theo `(updated_at, id)` hoặc `(created_at, id)`. Cursor sai định dạng trả 422 `INVALID_CURSOR`.
- `limit` tối đa 100 cho cả hai danh sách.

### 4.2 `UIMessage`

```json
{"id":"<uuid>","role":"assistant",
 "parts":[
   {"type":"text","text":"Người lớn … 0,5–1 g mỗi 4–6 giờ [1] …"},
   {"type":"source-document","sourceId":"<chunk_version_id>","mediaType":"text/markdown",
    "title":"Paracetamol › Liều lượng và cách dùng","providerMetadata":{"pharma":{…}}}
 ],
 "metadata":{"status":"completed","createdAt":"2026-09-13T08:12:00Z",
             "feedback":{"rating":"down","note":"…"}}}
```

- Tin nhắn người dùng chỉ có part `text`, `metadata = {status, createdAt}`.
- `metadata.feedback` là feedback hiện tại của người dùng cho tin nhắn đó hoặc `null`.
- Phase và evidence không có trong history.
- `source-document` dùng đúng hàm dựng như stream (§3.4). `providerMetadata.pharma.isCurrent` có mặt trong history, bằng `false` khi release của citation không còn là release hiện hành.
- Model `UIMessage`, các part và `MessageMetadata` là model Pydantic nằm trong `components` của OpenAPI.

## 5. Citation

### 5.1 Bảng `message_citations` (schema `public`)

| Cột | Kiểu | Ràng buộc |
| --- | --- | --- |
| `message_id` | uuid | FK `messages` `ON DELETE CASCADE` |
| `index` | int | PK `(message_id, index)` |
| `chunk_version_id` | uuid | FK `corpus.chunk_versions` `ON DELETE RESTRICT` |
| `release_id` | uuid | FK `corpus.releases` `ON DELETE RESTRICT` |
| `strategy` | text | `search_only \| chunk_window \| full_section` |
| `block_chunk_version_ids` | uuid[] | Thứ tự các chunk trong khối mà mô hình đã đọc |

Cột `messages.citations` (JSONB) bị bỏ. Môi trường dev được reset nên không backfill.

### 5.2 `GET /api/v1/messages/{message_id}/citations/{index}`

Trả `CitationDetail`:

```json
{"index":1,"source":"Dược thư Quốc gia Việt Nam (Xuất bản lần thứ 2) – …",
 "document_title":"Paracetamol","section":"Liều lượng và cách dùng",
 "start_page":812,"end_page":813,"strategy":"full_section","is_current":true,
 "chunks":[{"id":"<uuid>","text":"<markdown>","matched":false,"start_page":811,"end_page":811},
           {"id":"<uuid>","text":"<markdown>","matched":true,"start_page":812,"end_page":812}]}
```

`CitationDetail` là model REST nên dùng snake_case như các model REST khác (§3.2).

- `chunks` theo đúng thứ tự `block_chunk_version_ids`; `matched = true` cho `chunk_version_id` khớp truy vấn.
- Chỉ chủ hội thoại được đọc; không tồn tại hoặc không có quyền đều trả 404 `CITATION_NOT_FOUND`.

## 6. Auth

### 6.1 Hai auth backend (fastapi-users)

| Backend | Transport | Strategy | Route | Dùng cho |
| --- | --- | --- | --- | --- |
| `cookie` | `CookieTransport(cookie_name="pharma_session", cookie_httponly=True, cookie_secure=settings.auth.cookie_secure, cookie_samesite="lax", cookie_max_age=settings.auth.session_lifetime_seconds)` | `DatabaseStrategy(SQLAlchemyAccessTokenDatabase, lifetime_seconds=settings.auth.session_lifetime_seconds)` | `/auth/cookie/login`, `/auth/cookie/logout` | Trình duyệt |
| `jwt` | `BearerTransport` (giữ nguyên) | `JWTStrategy` (giữ nguyên) | `/auth/jwt/login`, `/auth/jwt/logout` | CLI, test |

- `FastAPIUsers(get_user_manager, [cookie_backend, jwt_backend])`; `current_active_user` chấp nhận cả hai.
- Bảng `access_tokens` từ `SQLAlchemyBaseAccessTokenTableUUID`, migration Alembic. Logout cookie xoá token trong DB.
- Lệnh `pharma-agent cleanup-sessions` xoá token hết hạn, theo mẫu `cleanup-checkpoints`.
- Register, `/users/me` (GET, PATCH), `/users/{id}` giữ nguyên.

### 6.2 CSRF

```python
CSRFMiddleware(
    secret=settings.auth.require_csrf_secret(),
    sensitive_cookies={"pharma_session"},
    cookie_name="csrftoken",
    header_name="x-csrftoken",
    cookie_secure=settings.auth.cookie_secure,
    cookie_samesite="lax",
)
```

- Chỉ request có cookie `pharma_session` bị kiểm tra; request bearer (CLI, test) không bị ảnh hưởng.
- Cookie `csrftoken` đọc được bằng JS; frontend gắn `x-csrftoken` cho mọi request không an toàn.
- Đã spike trên Starlette 1.6.0: bearer không có cookie session → 200; có cookie session, thiếu header → 403; có header → 200.

### 6.3 Google OAuth

- `get_oauth_router(google, cookie_backend, secret, redirect_url=f"{frontend_url}/auth/google/callback", associate_by_email=True, is_verified_by_default=True, csrf_token_cookie_secure=settings.auth.cookie_secure)`.
- Luồng: frontend gọi `GET /auth/google/authorize` → chuyển tới `authorization_url` → Google về trang frontend `/auth/google/callback?code&state` → frontend gọi `GET /api/v1/auth/google/callback?code&state` (cùng origin, cookie state tự gửi) → `backend.login()` trả 204 kèm `Set-Cookie: pharma_session`.

### 6.4 Setting

| Setting | Mặc định | Ghi chú |
| --- | --- | --- |
| `auth.session_lifetime_seconds` | `604800` | 7 ngày |
| `auth.cookie_secure` | `true` | Trình duyệt chấp nhận cookie Secure trên `http://localhost` khi dev |
| `auth.csrf_secret` | không có | Bắt buộc, tối thiểu 32 ký tự |
| `auth.frontend_url` | `http://localhost:3000` | Giữ nguyên, dùng cho OAuth |
| `api.cors_origins` | `[]` | Trình duyệt chỉ thấy một origin nên không cần CORS |

Ngoài phạm vi: quên mật khẩu và xác thực email (chưa có dịch vụ gửi email).

## 7. Lỗi theo RFC 9457

- Mọi lỗi trả `Content-Type: application/problem+json`:

```json
{"type":"urn:pharma-agent:problem:conversation-not-found","title":"Conversation not found",
 "status":404,"detail":"…","code":"CONVERSATION_NOT_FOUND",
 "errors":[{"loc":["body","message"],"message":"…","type":"string_too_long"}]}
```

- `errors` chỉ có ở 422.
- Exception handler của FastAPI:
  - `DomainError` và lỗi application → `code` sẵn có (`CONVERSATION_NOT_FOUND`, `MESSAGE_NOT_FOUND`, `CITATION_NOT_FOUND`, `SKILL_NOT_FOUND`, `SKILL_NAME_TAKEN`, `PAYLOAD_TOO_LARGE`, `INVALID_INPUT`, `INVALID_CURSOR`, `AGENT_UNAVAILABLE`…).
  - `HTTPException` của fastapi-users: `detail` là mã chuỗi (`REGISTER_USER_ALREADY_EXISTS`, `REGISTER_INVALID_PASSWORD`, `LOGIN_BAD_CREDENTIALS`, `OAUTH_*`) → đưa vào `code`.
  - `RequestValidationError` → 422 `VALIDATION_ERROR` kèm `errors`.
  - Lỗi CSRF 403 → `CSRF_FAILED`; 401 → `UNAUTHORIZED`.
- Response lỗi được khai báo trong OpenAPI để orval sinh type lỗi.

## 8. OpenAPI

- `FastAPI(generate_unique_id_function=lambda route: route.name)`; test bảo đảm mọi operationId duy nhất.
- Lệnh `pharma-agent export-openapi --output <path>` ghi schema mà không chạy server; frontend commit `frontend/openapi.json`. Test backend kiểm tra file export khớp app.
- `/chat/stream` khai báo response `text/event-stream`.
- Các model `UIMessage`, part, `MessageMetadata`, `EvidenceItem`, `CitationDetail`, `Problem` nằm trong `components`.

## 9. Server E2E

`backend/tests/e2e/server.py`, chạy `uv run python -m tests.e2e.server --port 8001`:

- App thật (`create_app`) với Postgres và Qdrant thật (service trong docker-compose), migration chạy khi khởi động.
- Import bundle fixture nhỏ của spec C bằng `ImportKnowledgeBundle` với embedder giả trả vector cố định theo hash.
- `FakeLlm` chạy kịch bản theo nội dung câu hỏi (ví dụ chứa `[e2e:blocked]`, `[e2e:timeout]`, mặc định là câu trả lời có citation `[1]`).
- Không có cờ "chế độ giả" trong code production; mọi thứ nằm trong `tests/`.

## 10. Testing

| Loại | Nội dung |
| --- | --- |
| Unit encoder | Chuỗi chunk mẫu cho từng `status`, có và không có evidence, citation, skill |
| API chat | Qua `httpx.ASGITransport`: parse SSE, header, `[DONE]`, lỗi trước stream là problem+json, bảo đảm §3.5 |
| Contract | Test backend ghi fixture `backend/tests/contract/fixtures/ui-stream/<scenario>.sse`; test frontend (spec B) validate bằng `uiMessageChunkSchema` và render thử |
| Phân trang | Nhiều tin nhắn cùng `created_at`: không trùng, không sót; cursor sai trả 422 |
| Citation | Endpoint chi tiết đúng thứ tự khối, `matched`, `isCurrent`; người khác đọc trả 404; FK chặn `gc` |
| Auth | Cookie login/logout (logout thu hồi token); bearer vẫn chạy; ba ca CSRF; OAuth callback đặt cookie (provider giả); `cleanup-sessions` |
| Lỗi | Dạng problem+json cho từng nhóm lỗi |
| OpenAPI | operationId duy nhất; file export khớp |

Chuẩn chung: pytest `filterwarnings = ["error"]`, pyrefly strict, ruff dùng chung, không ignore.

## 11. Tài liệu cần cập nhật

- `backend/README.md`: bảng route, luồng curl (tạo hội thoại, cookie login với CSRF hoặc bearer), dạng stream mới.
- Spec 2026-09-11 thêm dòng trạng thái trỏ sang spec này và spec C cho các phần bị thay thế.

## 12. Ngoài scope

- Tạo lại, sửa hoặc rẽ nhánh câu trả lời; resume stream sau khi mất kết nối.
- Quên mật khẩu, xác thực email.
- Endpoint AG-UI.
