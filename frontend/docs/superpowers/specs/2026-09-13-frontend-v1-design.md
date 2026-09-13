# Thiết kế frontend v1

Ngày: 2026-09-13. Trạng thái: đã duyệt qua brainstorming, chờ implementation plan.

Spec liên quan, cùng đợt:

- `backend/docs/superpowers/specs/2026-09-13-web-client-contract-design.md` (contract backend, gọi tắt là spec A).
- `backend/docs/superpowers/specs/2026-09-13-corpus-platform-design.md` (corpus platform, gọi tắt là spec C).

## 1. Mục tiêu và phạm vi

Web client cho AI agent tra cứu thuốc, dùng thư viện có sẵn thay vì tự dựng, và sẵn cho các trang corpus public có SEO sau này.

Bản v1 gồm:

| Phần | Nội dung |
| --- | --- |
| Auth | Đăng ký, đăng nhập email/mật khẩu, Google OAuth, đăng xuất |
| Chat | Danh sách hội thoại (phân trang, đổi tên, xoá); chat streaming hiển thị phase, skill, evidence; câu trả lời có trích dẫn `[n]`, xem trước và xem toàn văn nguồn; feedback 👍/👎 kèm ghi chú; tải tin nhắn cũ khi cuộn lên |
| Skills | Xem skill hệ thống và skill của mình; tải `SKILL.md`, bật/tắt, xoá |
| Cài đặt | Tên hiển thị, đổi mật khẩu, ngôn ngữ, giao diện sáng/tối/hệ thống |
| Landing | Trang giới thiệu public render phía server, có i18n và SEO |

Ràng buộc từ roadmap: sau này có trang corpus public cần SSR/SEO, nên framework và auth phải cho phép render phía server.

## 2. Stack

Phiên bản kiểm tra ngày 2026-09-12/13; plan sẽ pin chính xác. Cột cuối so với frontend của repo baas mà người dùng đang quen.

| Tầng | Lựa chọn | So với baas |
| --- | --- | --- |
| Framework | React Router 8.3 framework mode (người kế nhiệm Remix), SSR bật; React 19.2; Vite 8 | Khác: baas dùng Next.js |
| UI | shadcn CLI 4.21, style `base-nova` (Base UI 1.8), Tailwind 4, Toast của shadcn | Giống shadcn/Tailwind; Toast thay sonner vì bản Base UI của shadcn dùng Toast |
| Chat | AI SDK 7 (`ai`, `@ai-sdk/react` `useChat`); component chat của shadcn (`MessageScroller`, `Message`, `Bubble`, `Marker`, `Attachment`; `@shadcn/react` 0.3); component "elements" copy từ registry assistant-ui; Streamdown 2.6 (`@streamdown/code`, `@streamdown/math`) | Mới: baas chưa có chat AI dạng text |
| Dữ liệu | TanStack Query 5; client sinh bằng orval 8 từ `openapi.json` (react-query, zod, mock MSW) | Giống TanStack Query; orval thay client viết tay |
| Form | react-hook-form 7.88 + `@hookform/resolvers` 5.9 + zod 4 | Giống |
| i18n | i18next 26 + react-i18next 17 + remix-i18next 8; `vi` mặc định, `en` | Khác: next-intl chỉ dành cho Next.js |
| Theme | remix-themes 2.0 | Khác: next-themes chỉ dành cho Next.js |
| Ngôn ngữ, lint | TypeScript 7; Oxlint 1.82 type-aware (`oxlint-tsgolint`); Prettier 3.9 + `prettier-plugin-tailwindcss` | Khác: baas dùng Biome |
| Test | Vitest 5 (Node + Browser Mode), Testing Library, MSW 2, `@shadcn/helpers` 0.2, Playwright 1.63 | Giống Testing Library; Vitest thay Jest vì chạy thẳng trên Vite |
| Package manager | npm đi kèm Node 24 | Giống |
| Git hook | Hook local trong `.pre-commit-config.yaml` ở root repo | Khác: repo thesis đã dùng pre-commit cho Python |

## 3. Kiến trúc lúc chạy

```text
Browser ──► nginx (một origin)
             ├─ /api/*  ──► FastAPI :8000        (proxy_buffering off, gzip off, proxy_read_timeout 1h)
             └─ /*      ──► React Router server  (Node 24, react-router-serve)
                              └─ loader SSR ──► FastAPI (URL nội bộ, chuyển tiếp Cookie)

Dev: `react-router dev` (Vite) proxy /api → http://localhost:8000, không cần nginx
```

## 4. Cấu trúc thư mục

```text
frontend/
  app/
    root.tsx                 <html lang>, provider i18n/theme/QueryClient, Toaster, ErrorBoundary
    routes.ts                khai báo route
    entry.server.tsx         lấy ra bằng `react-router reveal`, bọc I18nextProvider
    app.css                  Tailwind, @source cho streamdown và plugin
    routes/
      public/landing.tsx
      auth/login.tsx, register.tsx, google-callback.tsx
      app/layout.tsx         clientMiddleware chặn chưa đăng nhập, shell sidebar
      app/chat.tsx
      app/skills.tsx
      app/settings.tsx
      actions/theme.ts, actions/locale.ts
      not-found.tsx
    features/                mỗi feature có components/, hooks/, lib/ và test đặt cạnh code
      auth/  chat/  citations/  conversations/  skills/  settings/  landing/
    components/ui/           shadcn, sinh bằng CLI
    components/elements/     component copy từ registry assistant-ui
    api/
      fetcher.ts             mutator của orval
      query-client.ts
      problem.ts             ApiError từ application/problem+json
      gen/                   output của orval (commit)
    i18n/
      config.ts, resources/vi/*.json, resources/en/*.json, types.d.ts
    lib/
  openapi.json               export từ backend (commit)
  orval.config.ts
  tests/e2e/                 Playwright
  Dockerfile
```

## 5. Route và cách render

| URL | Module | Render | Dữ liệu |
| --- | --- | --- | --- |
| `/`, `/en` | `public/landing.tsx` (hai route id) | SSR + `prerender` | `loader` chỉ lấy i18n; `meta` title/description/OG; `link hreflang` vi, en, x-default |
| `/login`, `/register` | `auth/*` | SSR phần khung, form chạy ở client | react-hook-form + schema zod do orval sinh |
| `/auth/google/callback` | `auth/google-callback.tsx` | Chỉ client | `clientLoader` gọi callback API rồi chuyển tới `next` hoặc `/chat` |
| `/chat`, `/chat/:conversationId` | `app/chat.tsx` trong `app/layout.tsx` | Chỉ client (`clientLoader` + `HydrateFallback`) | §10 |
| `/skills` | `app/skills.tsx` | Chỉ client | TanStack Query + orval |
| `/settings` | `app/settings.tsx` | Chỉ client | TanStack Query + orval |
| `*` | `not-found.tsx` | SSR | Có i18n |

## 6. Auth ở frontend

- **Chặn truy cập**: `app/layout.tsx` export `clientMiddleware` gọi `queryClient.ensureQueryData(getUsersCurrentUserQueryOptions())`; lỗi 401 thì `redirect("/login?next=<path>")`. Không dựa vào middleware phía server vì nó không chạy khi chuyển trang ở client.
- **Đăng nhập**: `POST /api/v1/auth/cookie/login` (form-urlencoded) → `invalidateQueries` user → chuyển tới `next`. Đã đăng nhập mà vào `/login` thì chuyển thẳng.
- **Google**: nút gọi `GET /auth/google/authorize` rồi chuyển tới `authorization_url`; trang callback gọi API callback cùng origin.
- **Đăng xuất**: `POST /api/v1/auth/cookie/logout` → `queryClient.clear()` → `/login`.
- **Hết phiên**: fetcher ném `ApiError(401)`; `QueryCache.onError` và `MutationCache.onError` chuyển về `/login?next=`.
- **CSRF**: fetcher và chat transport đọc cookie `csrftoken`, gắn `x-csrftoken` cho POST/PATCH/PUT/DELETE. Cookie có sẵn sau lần gọi `/users/me` ở bước chặn truy cập.

## 7. Tầng dữ liệu

### 7.1 orval

- Output 1: `client: "react-query"`, `httpClient: "fetch"`, `override.mutator` → `app/api/fetcher.ts`, `override.query.version: 5`, `useInfinite` cho các endpoint có `cursor` (`useInfiniteQueryParam: "cursor"`), `mock: true` sinh handler MSW.
- Output 2: `client: "zod"`, `version: "4"`, dùng cho form.
- Lệnh `npm run api:generate`; input là `frontend/openapi.json` do `pharma-agent export-openapi` ghi.
- Query key orval sinh ra là path tương đối cộng tham số, không chứa base URL, nên giống nhau ở server và client.

### 7.2 Fetcher

- Trình duyệt: base URL rỗng (cùng origin), `credentials: "same-origin"`, gắn CSRF.
- Server (khi trang public cần API): base URL `API_INTERNAL_URL`, chuyển tiếp `Cookie` qua tham số `fetch` của từng lần gọi. v1 chưa có trang public nào gọi API.
- Response `application/problem+json` chuyển thành `ApiError {status, code, detail, errors[]}`. Thông báo hiển thị lấy từ i18n theo `code`, thiếu thì dùng `title`.

### 7.3 QueryClient

- Route riêng tư: một instance duy nhất trên trình duyệt; `retry` bỏ qua lỗi 4xx.
- Trang public cần API (sau này): tạo instance mới cho mỗi request trên server, truyền sang client bằng `dehydrate` + `HydrationBoundary`.

### 7.4 Form

react-hook-form + `zodResolver(schema sinh bởi orval)`, giao diện theo hướng dẫn React Hook Form của shadcn (component `Field`). Lỗi 422 từ `errors[]` chuyển thành `form.setError(field, {message})`; lỗi khác vào `form.setError("root.server", …)`.

## 8. i18n

- remix-i18next middleware ở `root.tsx`; thứ tự xác định ngôn ngữ: `findLocale` theo URL có tiền tố `/en` (chỉ trang public) → cookie `lng` → `Accept-Language` → `vi`.
- Trang riêng tư không có tiền tố ngôn ngữ; chỉ trang public có `/en`.
- Bản dịch JSON đóng gói sẵn, namespace theo feature; key có type qua `CustomTypeOptions`.
- Nhãn tiếng Anh viết cứng trong component copy từ registry được thay bằng key i18n.
- `<html lang>` và `dir` theo ngôn ngữ hiện tại.

## 9. Giao diện

- `shadcn init -t react-router` với style `base-nova`, base color neutral, icon lucide, alias `~/components`.
- Component nền: `Sidebar`, `Toast`, `Sheet`, `Drawer`, `Dialog`, `AlertDialog`, `Card`, `Switch`, `Badge`, `InputGroup`, `Textarea`, `Field`.
- Dark mode: remix-themes với `createCookieSessionStorage` của React Router; `ThemeProvider` + `PreventFlashOnWrongTheme`; action `actions/theme.ts`. Đã kiểm tra: remix-themes chỉ import `react`, không phụ thuộc API của React Router, nên không bị ảnh hưởng bởi thay đổi của v8. Trang prerender dùng theme theo hệ thống.

## 10. Module chat

### 10.1 Thành phần

```text
routes/app/chat.tsx   (/chat, /chat/:conversationId)
├─ ConversationSidebar   Sidebar + useInfiniteQuery(listConversations); cuộn cuối thì tải thêm;
│                        đổi tên (Dialog), xoá (AlertDialog)
└─ ChatThread(conversationId)
   ├─ useInfiniteQuery(listMessages)       trang đầu, sau đó từng trang cũ hơn
   ├─ useChat<PharmaUIMessage>({ id, messages: trangĐầu, transport, onData,
   │                             dataPartSchemas, messageMetadataSchema })
   ├─ MessageScrollerProvider(autoScroll, defaultScrollPosition="last-anchor")
   │    MessageScrollerItem(messageId, scrollAnchor = role === "user") → Message/Bubble → MessageParts
   └─ Composer   InputGroup + Textarea; nút Gửi/Dừng theo status và stop(); tối đa 4000 ký tự
```

- `PharmaUIMessage = UIMessage<MessageMetadata, DataParts>`; type và schema zod lấy từ output orval (model `UIMessage`, `EvidenceItem`… trong OpenAPI), nên `dataPartSchemas` và `messageMetadataSchema` kiểm dữ liệu stream bằng đúng type của backend.

### 10.2 Hội thoại mới

1. Ở `/chat`, người dùng gửi tin nhắn đầu.
2. Gọi `POST /conversations`, rồi `navigate("/chat/<id>", {replace: true})` và gửi tin nhắn với `id` đó.
3. Part `data-conversation` (transient, nhận qua `onData`) cập nhật tiêu đề trong cache sidebar bằng `setQueryData`.

### 10.3 Transport

`DefaultChatTransport({ api: "/api/v1/chat/stream", credentials: "same-origin", headers: () => ({ "x-csrftoken": readCookie("csrftoken") }), prepareSendMessagesRequest: ({ id, messages }) => ({ body: { conversation_id: id, message: textOf(messages.at(-1)) } }) })`.

### 10.4 Tải tin nhắn cũ

- Khi `useMessageScrollerScrollable().start === false` (đã tới đầu) và còn `hasNextPage`: `fetchNextPage()` rồi `chat.setMessages(prev => [...older, ...prev])`.
- `MessageScrollerViewport` giữ `preserveScrollOnPrepend` (mặc định bật) nên vị trí đang đọc không nhảy.
- `messageId` của mọi item là ID từ backend (ổn định).

### 10.5 Hiển thị part và metadata

| Dữ liệu | Component | Hành vi |
| --- | --- | --- |
| `data-phase` | `AgentStatus` (elements) | Nhãn i18n, ví dụ "Đang tìm kiếm · vòng 2"; ẩn khi message xong |
| `data-skills` | `Badge` | Skill đã chọn |
| `data-evidence` | `RetrievalChunks` (elements) | Mở khi đang tìm, thu gọn thành "Đã đọc N đoạn" khi text bắt đầu; bỏ phần score |
| `text` | `Streamdown` | Đổi `[n]` thành `<cite-ref index="n"></cite-ref>` trước khi render; `allowedTags={{"cite-ref": ["index"]}}`, `literalTagContent={["cite-ref"]}`, `components={{"cite-ref": CitationMarker}}`; plugin `code`, `math`; `isAnimating` theo status |
| `source-document` | `Sources` (elements) | Nút "N nguồn", bấm mở danh sách |
| `metadata.status ∈ {blocked, redirected}` | `GuardrailNotice` (elements) | Thay kiểu hiển thị câu trả lời thường |
| `metadata.status ∈ {error, timeout}` | `ErrorState` (elements) | Nút "Thử lại" gửi lại đúng câu hỏi |
| `metadata.persisted === false` | Toast | "Không lưu được lượt này" |

- `CitationMarker` lấy từ hàm `Citation` bên trong `elements-inline-citation` (export ra, bỏ đoạn văn demo). Ô xem trước (Base UI `PreviewCard`) hiện nguồn, "Trích từ mục {title} › {section}", trang và `snippet`. Trên thiết bị cảm ứng, nếu `PreviewCard` không mở khi chạm thì dùng `Popover`.
- `providerMetadata.pharma.isCurrent === false` hiện dòng "Nguồn đã có phiên bản mới hơn".

### 10.6 Xem toàn văn citation

`CitationSheet` (`Sheet` trên desktop, `Drawer` trên mobile), mở khi bấm marker hoặc thẻ nguồn:

- `useQuery(getGetMessageCitationQueryOptions(messageId, index))` → `CitationDetail` (spec A §5.2).
- Header: nguồn, tên thuốc › mục, trang, strategy ("Toàn bộ mục", "Đoạn và các đoạn lân cận", "Một đoạn").
- Nội dung: từng chunk render bằng `Streamdown` chế độ tĩnh; chunk `matched` được tô nổi bật và tự cuộn tới.

### 10.7 Feedback

- Action bar: Copy, 👍, 👎; 👍/👎 là mutation orval `POST /messages/{id}/feedback {rating}`.
- Bấm 👎 mở `FeedbackDialog` (elements) để chọn lý do và ghi chú, rồi gửi lại (backend ghi đè).
- Trạng thái ban đầu lấy từ `metadata.feedback`; cập nhật lạc quan vào cache của `useChat` và query tin nhắn.

### 10.8 Lỗi và trạng thái

| Tình huống | Xử lý |
| --- | --- |
| 401 | Về `/login?next=` |
| 404 `CONVERSATION_NOT_FOUND` | Toast, về `/chat` |
| 422 | Composer kiểm độ dài trước khi gửi; nếu vẫn lọt thì hiện lỗi từ problem+json |
| 503 `AGENT_UNAVAILABLE` | Banner "Agent tạm thời không sẵn sàng" |
| Bấm Dừng | `stop()` huỷ fetch, backend huỷ graph; lượt không được lưu, tin nhắn ghi "Đã dừng" |
| Mất mạng giữa chừng | `onError` → `ErrorState` kèm thử lại |

Không làm: tạo lại, sửa, rẽ nhánh câu trả lời (backend không hỗ trợ).

### 10.9 Component copy từ registry

Cài qua shadcn CLI, bản Base UI, vào `app/components/elements/`: `elements-agent-status`, `elements-retrieval-chunks`, `elements-sources`, `elements-inline-citation`, `elements-feedback-dialog`, `elements-guardrail-notice`, `elements-error-state` (và `elements-surfaces`, `elements-range` đi kèm). Các component này chỉ phụ thuộc React, Base UI, lucide và `cn`, không cần runtime assistant-ui. Code được sửa trực tiếp trong repo (đổi field, thay nhãn bằng i18n).

## 11. Các màn hình khác

| Màn hình | Nội dung | Thành phần |
| --- | --- | --- |
| Landing `/`, `/en` | Giới thiệu, tính năng, nguồn dữ liệu (Dược thư Quốc gia, tờ hướng dẫn sử dụng), nút đăng nhập | SSR + prerender, meta SEO, hreflang; không gọi API |
| Đăng nhập, đăng ký | Email + mật khẩu, nút Google | react-hook-form + zod; lỗi `REGISTER_USER_ALREADY_EXISTS`, `LOGIN_BAD_CREDENTIALS` qua i18n |
| Skills | Skill hệ thống (chỉ xem), skill của mình: tải `SKILL.md`, bật/tắt, xoá | `Card`, `Switch`, `AlertDialog`, `Attachment`; lỗi 413/422/409 hiện ngay trên form tải lên |
| Cài đặt | Tên hiển thị, đổi mật khẩu (`PATCH /users/me`), ngôn ngữ (action ghi cookie `lng`), giao diện (action của remix-themes) | `Field`, `Select`, `RadioGroup` |
| Khung chung | Sidebar: hội thoại, Skills, Cài đặt, menu người dùng (đăng xuất) | `Sidebar`, thu gọn trên mobile |

## 12. Tooling

| Việc | Cấu hình |
| --- | --- |
| Scripts | `dev`, `build`, `start` (`react-router-serve ./build/server/index.js`), `typecheck` (`react-router typegen && tsc`), `lint` (`oxlint --type-aware --deny-warnings`), `format` (`prettier --check .`), `test` (Vitest), `e2e` (Playwright), `api:generate` (orval) |
| Oxlint | Plugin `typescript, react, jsx-a11y, import, vitest, promise`; nhóm `correctness`, `suspicious` báo lỗi; rule hook `react/rules-of-hooks`, `react/exhaustive-deps`. Không dùng `oxlint-disable` |
| Prettier | `plugins: ["prettier-plugin-tailwindcss"]`, `tailwindStylesheet: "./app/app.css"`, `tailwindFunctions: ["cn", "cva"]` |
| TypeScript | `strict`, `noUncheckedIndexedAccess`; `include` `.react-router/types/**/*`, `rootDirs` cho `+types`. Không dùng `@ts-ignore`/`@ts-expect-error` |
| Vitest | Hai project: `unit` (môi trường Node), `browser` (Browser Mode, Chromium qua `@vitest/browser-playwright`, `vitest-browser-react`). MSW `onUnhandledRequest: "error"`; warning hoặc error trên console làm test fail |
| Pre-commit ở root | Hook local cho `^frontend/`: oxlint, `prettier --check`, typecheck, và kiểm tra output orval khớp `openapi.json` |

Chuẩn chung với các project Python: lỗi lint/type sửa trong code, không tắt rule, warning tính là lỗi.

## 13. Testing

| Loại | Nội dung |
| --- | --- |
| Unit | Đổi `[n]` sang `cite-ref`; chèn tin nhắn cũ; chuyển problem+json thành `ApiError`; đọc cookie CSRF; mapping metadata |
| Component (Browser Mode + MSW) | Chat với stream dựng bằng `@shadcn/helpers` `createChat()` + `createUIMessageStreamResponse`: trả lời có citation, bị chặn, timeout, không lưu được, 401. Cuộn lên đầu tải tin nhắn cũ mà vị trí đọc giữ nguyên. Form đăng nhập, tải skill với lỗi 409/413/422 |
| Contract | Đọc fixture `backend/tests/contract/fixtures/ui-stream/*.sse` (spec A §10), validate từng chunk bằng `uiMessageChunkSchema` của `ai` và render không lỗi |
| E2E (Playwright) | `webServer` khởi động server E2E của backend (spec A §9) và `npm run build && npm start`. Kịch bản: đăng ký → đăng nhập → đăng xuất; hỏi → thấy phase → câu trả lời có `[1]` → xem trước → mở toàn văn; 👎 kèm ghi chú; tải, bật, tắt, xoá skill; tải tin nhắn cũ; đổi ngôn ngữ và giao diện; hết phiên về đăng nhập; `/en` có đúng `hreflang` |

## 14. Deploy

- `frontend/Dockerfile`: nhiều stage trên `node:24-alpine` (`npm ci`, `react-router build`, stage runtime chỉ cài dependency production), chạy `react-router-serve` với `PORT=3000`.
- `docker/nginx/nginx.conf`: `location /api/` proxy tới backend với `proxy_http_version 1.1`, `proxy_set_header Connection ""`, `proxy_buffering off`, `proxy_cache off`, `gzip off`, `proxy_read_timeout 1h`; `location /` proxy tới frontend.
- `docker-compose.yml`: thêm service `frontend` và `nginx`, `network_mode: host` giống backend, cổng lấy từ `.env` (`FRONTEND_PORT`, `WEB_PORT`).

## 15. Rủi ro đã biết

| Rủi ro | Cách xử lý |
| --- | --- |
| `@shadcn/react` 0.3, component chat mới từ 6/2026 | Pin phiên bản; test Browser Mode cho cuộn và chèn tin nhắn cũ |
| Component elements của assistant-ui mới, có đoạn demo | Code nằm trong repo, sửa trực tiếp; test component |
| `PreviewCard` trên thiết bị cảm ứng | Kiểm tra khi làm; không mở khi chạm thì dùng `Popover` |
| remix-themes không có release từ 1/2025 | Đã đọc mã nguồn: chỉ phụ thuộc React; bước đầu plan có test SSR theme trên v8 |
| TypeScript 7 chưa có API programmatic | Không dùng công cụ cần API đó (typescript-eslint, hey-api 0.99); orval và Oxlint đã kiểm tra chạy với TS 7 |

## 16. Ngoài scope

Trang corpus public và upload corpus cá nhân (roadmap); tạo lại, sửa, rẽ nhánh câu trả lời; quên mật khẩu; PWA; analytics.

## 17. Nhật ký quyết định

| Quyết định | Lựa chọn khác đã cân nhắc | Lý do |
| --- | --- | --- |
| React Router v8 framework mode | TanStack Start (vẫn RC); TanStack Router SPA (không SSR); Remix 3 (không còn là framework React) | Người kế nhiệm Remix, stable, cộng đồng lớn nhất, có SSR cho trang public |
| AI SDK `useChat` + UI Message Stream | AG-UI + assistant-ui; LangGraph `useStream` | Phổ biến nhất, protocol stable từ AI SDK v5; LangGraph `useStream` cần LangGraph Server |
| shadcn chat + `useChat`, không dùng runtime assistant-ui | Runtime assistant-ui (`useRemoteThreadListRuntime` + `useChatRuntime`); AI Elements của Vercel | assistant-ui không phân trang history và viewport không giữ vị trí khi chèn tin nhắn cũ; `MessageScroller` có sẵn `preserveScrollOnPrepend`; bớt một tầng state và các lỗi đã biết của adapter. AI Elements vẫn ghim `ai@6` |
| orval | @hey-api/openapi-ts (0.99 crash với TS 7; bản `next` là pre-release); openapi-typescript + openapi-fetch (không release 7 tháng) | Stable, chạy với TS 7, có zod và mock MSW |
| Oxlint + Prettier | Biome (rule type-aware và sắp xếp class Tailwind còn thử nghiệm); ESLint + Prettier (phải ở TS 6) | Lint type-aware stable trên TS 7; Prettier sắp xếp class Tailwind v4 ổn định |
| react-hook-form, npm | TanStack Form, pnpm | Giống baas, phổ biến hơn, người dùng quen |
| Vitest | Jest (như baas) | Chạy thẳng trên Vite, API gần như Jest |
| Cookie HttpOnly cùng origin | BFF trong React Router; JWT trong `localStorage` | Best practice cho web app, cho phép SSR, thu hồi được session |
| Base UI | Radix | Mặc định của shadcn từ 7/2026, được chọn nhiều gấp đôi trên `shadcn create` |
| i18next + remix-i18next | Lingui, Paraglide | Phổ biến nhất, hỗ trợ middleware React Router v8 |
