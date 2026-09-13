# Frontend Chat Implementation Plan (Plan 9 of 10)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the chat module of the frontend (spec B §10): streaming chat on the AI SDK UI Message Stream with phase, skills, evidence, answers with `[n]` citations (hover preview, full-text sheet), feedback, a conversation list with paging, rename and delete, loading older messages while keeping the reading position, and the error handling of §10.8.

**Architecture:** `routes/app/chat.tsx` (replacing P8's placeholder inside P8's private layout) renders `ChatThread`. `ChatThread` loads the first history page with the orval infinite query, then mounts `ChatSession`, which owns `useChat<PharmaUIMessage>` with a `DefaultChatTransport` that posts `{conversation_id, message}`. Messages render inside shadcn `MessageScroller`; a pure view mapper (`describeMessage`) feeds `MessageParts`, which uses copied assistant-ui "elements" (Base UI) and Streamdown. Citations flow through a React context so the Streamdown `cite-ref` component shows a `PreviewCard` and opens `CitationSheet`. `ConversationSidebar` is mounted in P8's `AppSidebar`. Pure logic lives in `lib/` files with Node unit tests; components have Vitest Browser Mode tests with MSW, where streams are scripted with `@shadcn/helpers` `createChat()` and served through the real `DefaultChatTransport`. A contract test validates and renders backend P6's `.sse` fixtures.

**Tech Stack:** as P8 (React Router 8.3.1, React 19.3, TypeScript 7.0.2, TanStack Query 5.102.8, orval 8.32.0 output, zod 4.6.4, react-hook-form 7.88.0, `@hookform/resolvers` 5.9.1, i18next 26.4.2, react-i18next 17.0.13, shadcn 4.21.0 `base-nova`, `@base-ui/react` 1.8.0, lucide-react 1.45.0, Vitest 5.0.0, `vitest-browser-react` 2.3.0, MSW 2.15.0) plus `ai` 7.0.99, `@ai-sdk/react` 4.0.102, `@shadcn/react` 0.3.1 (through the `message-scroller` registry item), assistant-ui registry "elements" (`https://r.assistant-ui.com/<name>.json`, one file for every style), `tw-shimmer` 0.4.13, `streamdown` 2.6.0, `@streamdown/code` 1.1.1, `@streamdown/math` 1.0.2, `katex` (the version `@streamdown/math` resolves), `react-intersection-observer` 11.0.1, `@shadcn/helpers` 0.2.0 (dev).

**Spec:** `frontend/docs/superpowers/specs/2026-09-13-frontend-v1-design.md` §10 (all), §13 (chat unit, component and contract rows); backend contract `backend/docs/superpowers/specs/2026-09-13-web-client-contract-design.md` §3–§5, §7. Cross-plan names: `backend/docs/superpowers/plans/2026-09-13-plans-overview.md` §3.5 and §4. Builds on P8 `frontend/docs/superpowers/plans/2026-09-13-frontend-foundation.md` (Tasks 1–11), backend P5 `2026-09-13-web-api-foundation.md` and P6 `2026-09-13-web-chat-stream.md`.

## Global Constraints

- TypeScript strict with `noUncheckedIndexedAccess`; Oxlint type-aware with `--deny-warnings` and P8's `.oxlintrc.json` (categories `correctness` and `suspicious`, including `typescript/no-unsafe-type-assertion`, `typescript/no-floating-promises`, `vitest/require-mock-type-parameters`, `eslint/no-shadow`, React Compiler rules such as `react/set-state-in-effect`). No `oxlint-disable`, `@ts-ignore`, `@ts-expect-error`, and no `as T` casts (only `as const`). Network data is typed by orval or parsed with zod.
- Prettier is P8's config (`semi: false`, double quotes, `trailingComma: "es5"`, width 80, Tailwind class sorting). Code blocks below are written without semicolons; every full check runs `npm run format:write` first so line wrapping and class order match.
- Every task ends green on `npm run format:write && npm run lint && npm run format && npm run typecheck && npm test` (run from `frontend/`). A `console.error`/`console.warn` during a test and an unhandled `/api/` request fail the run (P8 `tests/setup/`).
- Product copy comes from i18n keys (`chat`, `citations`, `errors`); Vietnamese is the default. `vi` and `en` define identical key sets (P8 `app/i18n/resources.test.ts` compares them exactly, so plural keys exist as `_one` and `_other` in both). No medical disclaimers.
- Prefer library behaviour over custom code: `useChat` state, `DefaultChatTransport`, `MessageScroller` prepend preservation, Streamdown, TanStack Query caches, `react-intersection-observer`. Registry components are copied into the repo and edited there.
- Test support lives under `tests/` (P8 convention). No test-only flags in production code.
- One commit per task, conventional message. Every commit message ends with the session attribution trailer of the executing session; the executing session sets `COMMIT_TRAILER` to that trailer, and commit steps pass it as the last `-m` paragraph.

### Names consumed from P8, P5 and P6

| Source | Names |
| --- | --- |
| P8 test support | `worker` from `tests/msw/browser.ts`; `createTestI18n(language?)` from `tests/utils/i18n.ts` (synchronous, all namespaces); `TestProviders({ children, client?, language? })` and `createTestQueryClient()` from `tests/utils/providers.tsx`; browser tests `app/**/*.browser.test.{ts,tsx}`, unit tests `app/**/*.test.{ts,tsx}` |
| P8 app | `toast` (`toast.add({ title, description, type })`) and `Toaster` from `~/components/ui/toast`; `useIsMobile()` from `~/hooks/use-mobile`; `ApiError` (constructor `{ status, code, title, detail?, errors? }`), `isApiError` from `~/api/problem`; `apiErrorMessage(error, t)` from `~/i18n/error-message`; `readCsrfToken`, `CSRF_HEADER` from `~/lib/csrf`; shadcn `button`, `sidebar`, `sheet`, `drawer`, `dialog`, `alert-dialog`, `badge`, `input-group`, `textarea`, `field`, `input`, `label`, `dropdown-menu`, `skeleton`, `spinner`, `tooltip`; `AppSidebar` in `app/features/shell/AppSidebar.tsx`; private layout `routes/app/layout.tsx` (header `h-12` above `<Outlet />`); route `chat/:conversationId?` → `routes/app/chat.tsx` (placeholder replaced here) |
| P8 orval output | `~/api/gen/endpoints` (`useListConversationsInfinite`, `getListConversationsInfiniteQueryKey`, `useListMessagesInfinite`, `getListMessagesInfiniteQueryKey`, `useCreateConversation` with variables `void`, `useRenameConversation` `{ conversationId, data }`, `useDeleteConversation` `{ conversationId }`, `getGetMessageCitationQueryOptions(messageId, index)`, `useSubmitFeedback` `{ messageId, data }`); `~/api/gen/endpoints.msw` (`getListMessagesMockHandler`, `getListConversationsMockHandler`, `getCreateConversationMockHandler`, `getGetMessageCitationMockHandler`); `~/api/gen/schemas`; `~/api/gen/zod` (`ListMessagesResponse`, `RenameConversationBody`, `chatStreamBodyMessageMax`, `submitFeedbackBodyNoteMax`) |
| P5 | `POST /conversations` has no body (201 `ConversationView {id, title, turn_count, created_at, updated_at}`); `GET /conversations?limit=20&cursor=` → `ConversationPage`; `GET /conversations/{id}/messages?limit=30&cursor=` → `MessagePage` (items oldest first, `next_cursor` older); problem+json with `code`; codes shown here: `UNAUTHORIZED`, `CONVERSATION_NOT_FOUND`, `CITATION_NOT_FOUND`, `MESSAGE_NOT_FOUND`, `VALIDATION_ERROR`, `AGENT_UNAVAILABLE` (any 503, including `SERVICE_STARTING`, shows the unavailable banner) |
| P6 | Schemas `UIMessage`, `TextUIPart`, `SourceDocumentUIPart`, `MessageMetadata {status, createdAt, errorCode?, usage?, runId?, persisted?, feedback?}`, `MessageFeedback`, `PharmaSourceMetadata {index, source, title, section, startPage, endPage, snippet, isCurrent}`, `EvidenceItem`, `PharmaDataParts {phase: PhaseData, skills: SkillsData, evidence: EvidenceData, conversation: ConversationData}`, `CitationDetail` (snake_case, `strategy: HydrateStrategy`, `is_current`), `CitationChunk`; chunks: `start` → `data-conversation` (transient) → `data-phase` (id `phase`, `round` only while searching) → `data-skills` → `data-evidence` (id `evidence`) → `text-start/delta/end` (id `text`) → `source-document` per citation → `finish` (`finishReason` `error` for `error`/`timeout`) → `data: [DONE]`; fixtures `backend/tests/contract/fixtures/ui-stream/{completed-with-citations,blocked,timeout,persist-failed,no-evidence}.sse` |
| P10 contracts table | composer textbox `Tin nhắn`, button `Gửi`; phase text starting `Đang tìm kiếm`; citation markers are buttons named by their index (`1`); preview contains `Trích từ mục`; a marker opens a dialog whose header shows `Toàn bộ mục` / `Đoạn và các đoạn lân cận` / `Một đoạn`; action bar buttons `Hữu ích`, `Chưa hữu ích`; `FeedbackDialog` textbox `Ghi chú`, button `Gửi phản hồi`; each message item carries `[data-message-id]` |

`openapi.json` puts the stream's data-part schema under `text/event-stream`, where orval's zod client emits `zod.unknown()`. The data-part types therefore come from `~/api/gen/schemas`, and their zod schemas are written in `message-schema.ts` with two-way compile-time checks against those types (Task 3).

---

## File Structure

```text
frontend/
  package.json, package-lock.json                   (+ ai, @ai-sdk/react, streamdown, plugins, katex,
                                                      react-intersection-observer, @shadcn/react, tw-shimmer,
                                                      @shadcn/helpers)                                          # Task 2
  vitest.config.ts                                   (+ server.fs.allow for backend fixtures)                   # Task 12
  tests/chat/
    render.tsx                                       renderRoutes, LocationProbe                                # Task 1
    fixtures.ts                                      history, conversation and citation builders                # Task 1
    stream-handler.ts                                MSW handlers serving @shadcn/helpers chats and problems    # Task 9
    ui-stream-fixtures.ts                            loads and parses backend .sse fixtures                     # Task 12
  app/
    app.css                                          (+ Streamdown styles and @source lines)                    # Task 2
    i18n/resources/vi/chat.json, en/chat.json        chat copy                                                  # Task 1
    i18n/resources/vi/citations.json, en/citations.json citation copy                                           # Task 1
    routes/app/chat.tsx                              replaces P8 placeholder: clientLoader, HydrateFallback      # Task 9
    components/ui/                                   message-scroller, message, bubble, marker, collapsible,
                                                     alert, empty (CLI)                                         # Task 2
    components/elements/                             surfaces, agent-status, retrieval-chunks, sources,
                                                     inline-citation, feedback-dialog, guardrail-notice,
                                                     error-state (adapted from assistant-ui registry)           # Task 2
    features/shell/AppSidebar.tsx                    (+ ConversationSidebar)                                    # Task 11
    features/shell/AppSidebar.browser.test.tsx       (+ conversations handler)                                  # Task 11
    features/chat/
      ChatThread.tsx                                 first history page, NewChat or ChatSession                 # Task 9, 10
      test-support.browser.test.tsx                  checks the chat test helpers and copy                      # Task 1
      components/ChatSession.tsx                     useChat, MessageScroller, paging, error table, stop        # Task 9
      components/Composer.tsx                        InputGroup + Textarea, send/stop, 4000 chars               # Task 9
      components/OlderMessagesLoader.tsx             loads older pages at the top                               # Task 9
      components/AgentUnavailableBanner.tsx          503 banner                                                 # Task 9
      components/NewChat.tsx                         empty state, POST /conversations, navigate                 # Task 10
      components/MessageParts.tsx                    spec §10.5 rendering table                                 # Task 5
      components/EvidencePanel.tsx                   RetrievalChunks wrapper                                    # Task 5
      components/SourcesList.tsx                     Sources wrapper                                            # Task 5
      components/AnswerMarkdown.tsx                  Streamdown with cite-ref                                   # Task 4
      components/MessageActions.tsx                  copy, Hữu ích, Chưa hữu ích, FeedbackDialog                # Task 8
      lib/message-schema.ts                          PharmaUIMessage, data part and metadata schemas            # Task 3
      lib/chat-messages.ts                           history mapping and merge helpers                          # Task 3
      lib/cite-markers.ts                            toCiteRefMarkup, parseCiteIndex                            # Task 4
      lib/sources.ts                                 readPharmaSource, citationSourcesOf                        # Task 4
      lib/streamdown-config.ts                       shared Streamdown plugins                                  # Task 4
      lib/message-view.ts                            describeMessage                                            # Task 5
      lib/chat-transport.ts                          createChatTransport                                        # Task 6
      lib/stream-problem.ts                          readStreamProblem → ApiError                               # Task 6
      lib/feedback-note.ts                           composeFeedbackNote                                        # Task 8
      lib/location-state.ts                          chat navigation state schema                               # Task 9
      contract/ui-stream.test.ts, ui-stream.browser.test.tsx                                                   # Task 12
      *.test.ts, *.browser.test.tsx                  next to the code
    features/citations/
      CitationMarker.tsx                             cite-ref component with PreviewCard                        # Task 4
      citation-context.tsx                           MessageCitationsProvider, useMessageCitations              # Task 4
      lib/citation-format.ts                         pageLabel, usePageText                                     # Task 4
      CitationSheet.tsx                              Sheet (desktop) / Drawer (mobile), CitationDetailView      # Task 7
    features/conversations/
      lib/conversation-cache.ts                      title, rename, delete and feedback cache updates           # Task 8
      ConversationSidebar.tsx                        infinite list, load more, rename, delete                   # Task 11
      components/RenameConversationDialog.tsx                                                                    # Task 11
      components/DeleteConversationDialog.tsx                                                                    # Task 11
```

---

### Task 1: Chat copy and chat test support

**Files:**
- Modify (P8 created them as `{}`): `frontend/app/i18n/resources/vi/chat.json`, `frontend/app/i18n/resources/en/chat.json`, `frontend/app/i18n/resources/vi/citations.json`, `frontend/app/i18n/resources/en/citations.json`
- Create: `frontend/tests/chat/render.tsx`, `frontend/tests/chat/fixtures.ts`
- Test: `frontend/app/features/chat/test-support.browser.test.tsx`

**Interfaces:**
- Consumes: `TestProviders`, `createTestQueryClient` (`tests/utils/providers.tsx`); `Toaster`, `toast` (`~/components/ui/toast`); orval types from `~/api/gen/schemas`.
- Produces:
  - `tests/chat/render.tsx`: `LocationProbe()` (renders `<output data-testid="location">` with `pathname + search`); `renderWithProviders(ui: ReactNode): Promise<RenderResult>` (`TestProviders` + `Toaster`); `renderRoutes(Page: ComponentType, options: RenderRoutesOptions): Promise<RenderRoutesResult>` with `type RenderRoutesOptions = { initialEntry: string | { pathname: string; state: unknown }; path?: string; wrap?: ComponentType<{ children: ReactNode }> }` and `type RenderRoutesResult = { screen: RenderResult; queryClient: QueryClient }` (routes: `path` default `/chat/:conversationId?` rendering `Page` and `LocationProbe`, plus `/login` rendering `LocationProbe`).
  - `tests/chat/fixtures.ts`: `CREATED_AT`, `pharmaSource(index, overrides?)`, `apiUserMessage(id, text)`, `apiAssistantMessage(id, text, sources?, metadata?)`, `conversationTurns(count, prefix)`, `messagePage(items, nextCursor?)`, `conversationView(id, title, overrides?)`, `conversationPage(items, nextCursor?)`, `citationDetail(index, overrides?)`.
  - `chat` keys: `thread.*`, `newChat.*`, `composer.*`, `phase.*`, `status.*`, `skills.label`, `evidence.{searching,read_one,read_other}`, `sources.{trigger_one,trigger_other,stale}`, `citation.excerpt`, `guardrail.*`, `error.*`, `stopped`, `notPersisted`, `agentUnavailable.*`, `actions.*`, `feedback.*`, `conversations.*`. `citations` keys: `sheet.*`, `strategy.*`, `pages.*`, `stale`.

Accessible names follow the P10 contracts table: composer `Tin nhắn`, `Gửi`, `Hữu ích`, `Chưa hữu ích`, `Ghi chú`, `Gửi phản hồi`. Error messages come from P8's `errors` namespace (it already holds every code this plan shows).

- [ ] **Step 1: Write the failing test**

`frontend/app/features/chat/test-support.browser.test.tsx`:

```tsx
import { useTranslation } from "react-i18next"
import { useNavigate } from "react-router"
import { describe, expect, test } from "vitest"
import { page } from "vitest/browser"

import { toast } from "~/components/ui/toast"

import {
  apiAssistantMessage,
  conversationView,
  messagePage,
  pharmaSource,
} from "../../../tests/chat/fixtures"
import { renderRoutes } from "../../../tests/chat/render"

function Probe() {
  const { t } = useTranslation("chat")
  const navigate = useNavigate()
  return (
    <div>
      <button type="button" onClick={() => void navigate("/login?next=%2Fchat")}>
        {t("composer.send")}
      </button>
      <button type="button" onClick={() => toast.add({ title: t("notPersisted") })}>
        {t("composer.stop")}
      </button>
    </div>
  )
}

describe("chat test support", () => {
  test("renders chat copy in a router with toasts and reports the location", async () => {
    const { queryClient } = await renderRoutes(Probe, { initialEntry: "/chat/c1" })
    await expect.element(page.getByTestId("location")).toHaveTextContent("/chat/c1")
    await page.getByRole("button", { name: "Dừng" }).click()
    await expect.element(page.getByText("Không lưu được lượt này")).toBeVisible()
    await page.getByRole("button", { name: "Gửi" }).click()
    await expect
      .element(page.getByTestId("location"))
      .toHaveTextContent("/login?next=%2Fchat")
    expect(queryClient.getDefaultOptions().queries?.retry).toBe(false)
  })

  test("fixtures follow the generated API types", () => {
    const history = messagePage(
      [apiAssistantMessage("a1", "Liều [1].", [pharmaSource(1)])],
      "cursor-1"
    )
    expect(history.next_cursor).toBe("cursor-1")
    expect(history.items[0]?.parts[1]?.type).toBe("source-document")
    expect(conversationView("c1", "Paracetamol").turn_count).toBe(1)
  })
})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd frontend && npx vitest run --project browser app/features/chat/test-support.browser.test.tsx`
Expected: FAIL with `Failed to resolve import "../../../tests/chat/fixtures"`.

- [ ] **Step 3: Write the copy**

`frontend/app/i18n/resources/vi/chat.json`:

```json
{
  "thread": {
    "label": "Tin nhắn trong cuộc trò chuyện",
    "loading": "Đang tải tin nhắn…",
    "loadingOlder": "Đang tải tin nhắn cũ…",
    "loadFailed": "Không tải được tin nhắn",
    "reload": "Tải lại",
    "scrollToEnd": "Cuộn xuống cuối"
  },
  "newChat": {
    "title": "Hỏi về thuốc",
    "description": "Hỏi về liều dùng, tương tác, chống chỉ định hoặc cách dùng thuốc.",
    "createFailed": "Không tạo được cuộc trò chuyện"
  },
  "composer": {
    "label": "Tin nhắn",
    "placeholder": "Nhập câu hỏi về thuốc…",
    "send": "Gửi",
    "stop": "Dừng",
    "tooLong": "Câu hỏi dài {{count}}/{{max}} ký tự"
  },
  "phase": {
    "guarding": "Đang kiểm tra câu hỏi",
    "understanding": "Đang phân tích câu hỏi",
    "selecting_skills": "Đang chọn kỹ năng",
    "searching": "Đang tìm kiếm",
    "searchingRound": "Đang tìm kiếm · vòng {{round}}",
    "reading": "Đang đọc tài liệu",
    "answering": "Đang trả lời"
  },
  "status": {
    "working": "Đang xử lý",
    "done": "Hoàn tất"
  },
  "skills": {
    "label": "Kỹ năng đã chọn"
  },
  "evidence": {
    "searching": "Đang tìm tài liệu",
    "read_one": "Đã đọc {{count}} đoạn",
    "read_other": "Đã đọc {{count}} đoạn"
  },
  "sources": {
    "trigger_one": "{{count}} nguồn",
    "trigger_other": "{{count}} nguồn",
    "stale": "Nguồn đã có phiên bản mới hơn"
  },
  "citation": {
    "excerpt": "Trích từ mục {{title}} › {{section}}"
  },
  "guardrail": {
    "blocked": "Không thể trả lời câu hỏi này",
    "redirected": "Câu hỏi nằm ngoài phạm vi tra cứu thuốc"
  },
  "error": {
    "title": "Chưa trả lời được",
    "timeout": "Agent phản hồi quá lâu.",
    "failed": "Có lỗi khi tạo câu trả lời.",
    "network": "Mất kết nối trong lúc trả lời.",
    "retry": "Thử lại",
    "retrying": "Đang thử lại"
  },
  "stopped": "Đã dừng",
  "notPersisted": "Không lưu được lượt này",
  "agentUnavailable": {
    "title": "Agent tạm thời không sẵn sàng",
    "description": "Vui lòng thử lại sau ít phút."
  },
  "actions": {
    "copy": "Sao chép",
    "copied": "Đã sao chép",
    "up": "Hữu ích",
    "down": "Chưa hữu ích",
    "feedbackFailed": "Không gửi được đánh giá"
  },
  "feedback": {
    "title": "Câu trả lời chưa tốt ở đâu?",
    "optional": "không bắt buộc",
    "notePlaceholder": "Ghi chú",
    "submit": "Gửi phản hồi",
    "sent": "Cảm ơn bạn đã góp ý.",
    "reasons": {
      "incorrect": "Sai thông tin",
      "incomplete": "Thiếu thông tin",
      "wrongSource": "Trích dẫn không khớp",
      "unclear": "Khó hiểu"
    }
  },
  "conversations": {
    "title": "Cuộc trò chuyện",
    "new": "Cuộc trò chuyện mới",
    "empty": "Chưa có cuộc trò chuyện nào",
    "loadFailed": "Không tải được danh sách",
    "more": "Thao tác với {{title}}",
    "rename": "Đổi tên",
    "delete": "Xoá",
    "renameTitle": "Đổi tên cuộc trò chuyện",
    "titleLabel": "Tên",
    "save": "Lưu",
    "cancel": "Huỷ",
    "deleteTitle": "Xoá cuộc trò chuyện?",
    "deleteDescription": "“{{title}}” và toàn bộ tin nhắn sẽ bị xoá vĩnh viễn.",
    "deleteConfirm": "Xoá",
    "renameFailed": "Không đổi tên được",
    "deleteFailed": "Không xoá được"
  }
}
```

`frontend/app/i18n/resources/en/chat.json`:

```json
{
  "thread": {
    "label": "Conversation messages",
    "loading": "Loading messages…",
    "loadingOlder": "Loading older messages…",
    "loadFailed": "Could not load messages",
    "reload": "Reload",
    "scrollToEnd": "Scroll to the end"
  },
  "newChat": {
    "title": "Ask about a medicine",
    "description": "Ask about dosage, interactions, contraindications or how to take a medicine.",
    "createFailed": "Could not start a conversation"
  },
  "composer": {
    "label": "Message",
    "placeholder": "Type a question about a medicine…",
    "send": "Send",
    "stop": "Stop",
    "tooLong": "Question is {{count}}/{{max}} characters"
  },
  "phase": {
    "guarding": "Checking the question",
    "understanding": "Understanding the question",
    "selecting_skills": "Choosing skills",
    "searching": "Searching",
    "searchingRound": "Searching · round {{round}}",
    "reading": "Reading documents",
    "answering": "Answering"
  },
  "status": {
    "working": "Working",
    "done": "Done"
  },
  "skills": {
    "label": "Selected skills"
  },
  "evidence": {
    "searching": "Searching documents",
    "read_one": "Read {{count}} passage",
    "read_other": "Read {{count}} passages"
  },
  "sources": {
    "trigger_one": "{{count}} source",
    "trigger_other": "{{count}} sources",
    "stale": "A newer version of this source exists"
  },
  "citation": {
    "excerpt": "From {{title}} › {{section}}"
  },
  "guardrail": {
    "blocked": "This question cannot be answered",
    "redirected": "This question is outside medicine lookup"
  },
  "error": {
    "title": "No answer yet",
    "timeout": "The agent took too long to respond.",
    "failed": "Something went wrong while answering.",
    "network": "The connection dropped while answering.",
    "retry": "Retry",
    "retrying": "Retrying"
  },
  "stopped": "Stopped",
  "notPersisted": "This turn could not be saved",
  "agentUnavailable": {
    "title": "The agent is temporarily unavailable",
    "description": "Please try again in a few minutes."
  },
  "actions": {
    "copy": "Copy",
    "copied": "Copied",
    "up": "Helpful",
    "down": "Not helpful",
    "feedbackFailed": "Could not send feedback"
  },
  "feedback": {
    "title": "What went wrong?",
    "optional": "optional",
    "notePlaceholder": "Note",
    "submit": "Send feedback",
    "sent": "Thanks for the feedback.",
    "reasons": {
      "incorrect": "Incorrect",
      "incomplete": "Incomplete",
      "wrongSource": "Citation does not match",
      "unclear": "Hard to understand"
    }
  },
  "conversations": {
    "title": "Conversations",
    "new": "New conversation",
    "empty": "No conversations yet",
    "loadFailed": "Could not load conversations",
    "more": "Actions for {{title}}",
    "rename": "Rename",
    "delete": "Delete",
    "renameTitle": "Rename conversation",
    "titleLabel": "Name",
    "save": "Save",
    "cancel": "Cancel",
    "deleteTitle": "Delete conversation?",
    "deleteDescription": "“{{title}}” and all its messages will be deleted permanently.",
    "deleteConfirm": "Delete",
    "renameFailed": "Could not rename",
    "deleteFailed": "Could not delete"
  }
}
```

`frontend/app/i18n/resources/vi/citations.json`:

```json
{
  "sheet": {
    "title": "Nguồn [{{index}}]",
    "loading": "Đang tải nội dung nguồn…",
    "loadFailed": "Không tải được nội dung nguồn",
    "matched": "Đoạn được trích dẫn"
  },
  "strategy": {
    "full_section": "Toàn bộ mục",
    "chunk_window": "Đoạn và các đoạn lân cận",
    "search_only": "Một đoạn"
  },
  "pages": {
    "single": "Trang {{page}}",
    "range": "Trang {{start}}–{{end}}"
  },
  "stale": "Nguồn đã có phiên bản mới hơn"
}
```

`frontend/app/i18n/resources/en/citations.json`:

```json
{
  "sheet": {
    "title": "Source [{{index}}]",
    "loading": "Loading the source…",
    "loadFailed": "Could not load the source",
    "matched": "Cited passage"
  },
  "strategy": {
    "full_section": "Whole section",
    "chunk_window": "Passage and nearby passages",
    "search_only": "Single passage"
  },
  "pages": {
    "single": "Page {{page}}",
    "range": "Pages {{start}}–{{end}}"
  },
  "stale": "A newer version of this source exists"
}
```

- [ ] **Step 4: Write the test support**

`frontend/tests/chat/render.tsx`:

```tsx
import type { QueryClient } from "@tanstack/react-query"
import type { ComponentType, ReactNode } from "react"
import { createRoutesStub, useLocation } from "react-router"
import { render, type RenderResult } from "vitest-browser-react"

import { Toaster } from "~/components/ui/toast"

import { createTestQueryClient, TestProviders } from "../utils/providers"

export type RenderRoutesOptions = {
  initialEntry: string | { pathname: string; state: unknown }
  path?: string
  wrap?: ComponentType<{ children: ReactNode }>
}

export type RenderRoutesResult = {
  screen: RenderResult
  queryClient: QueryClient
}

export function LocationProbe() {
  const location = useLocation()
  return (
    <output data-testid="location">{`${location.pathname}${location.search}`}</output>
  )
}

function PassThrough({ children }: { children: ReactNode }) {
  return <>{children}</>
}

export function renderWithProviders(ui: ReactNode): Promise<RenderResult> {
  return render(
    <TestProviders>
      <Toaster>{ui}</Toaster>
    </TestProviders>
  )
}

export async function renderRoutes(
  Page: ComponentType,
  {
    initialEntry,
    path = "/chat/:conversationId?",
    wrap: Wrap = PassThrough,
  }: RenderRoutesOptions
): Promise<RenderRoutesResult> {
  const queryClient = createTestQueryClient()

  function RoutePage() {
    return (
      <Wrap>
        <Page />
        <LocationProbe />
      </Wrap>
    )
  }

  const Stub = createRoutesStub([
    { path, Component: RoutePage },
    { path: "/login", Component: LocationProbe },
  ])

  const screen = await render(
    <TestProviders client={queryClient}>
      <Toaster>
        <Stub initialEntries={[initialEntry]} />
      </Toaster>
    </TestProviders>
  )
  return { screen, queryClient }
}
```

`frontend/tests/chat/fixtures.ts`:

```ts
import type {
  UIMessage as ApiUIMessage,
  CitationDetail,
  ConversationPage,
  ConversationView,
  MessageMetadata,
  MessagePage,
  PharmaSourceMetadata,
} from "~/api/gen/schemas"

export const CREATED_AT = "2026-09-13T08:00:00Z"

export function pharmaSource(
  index: number,
  overrides: Partial<PharmaSourceMetadata> = {}
): PharmaSourceMetadata {
  return {
    index,
    source: "Dược thư Quốc gia Việt Nam (Xuất bản lần thứ 2)",
    title: "Paracetamol",
    section: "Liều lượng và cách dùng",
    startPage: 812,
    endPage: 813,
    snippet: "Người lớn và trẻ em trên 12 tuổi uống 0,5–1 g mỗi 4–6 giờ.",
    isCurrent: true,
    ...overrides,
  }
}

export function apiUserMessage(id: string, text: string): ApiUIMessage {
  return {
    id,
    role: "user",
    parts: [{ type: "text", text }],
    metadata: { status: "completed", createdAt: CREATED_AT },
  }
}

export function apiAssistantMessage(
  id: string,
  text: string,
  sources: readonly PharmaSourceMetadata[] = [],
  metadata: Partial<MessageMetadata> = {}
): ApiUIMessage {
  return {
    id,
    role: "assistant",
    parts: [
      { type: "text", text },
      ...sources.map((source) => ({
        type: "source-document" as const,
        sourceId: `chunk-${source.index}`,
        mediaType: "text/markdown" as const,
        title: `${source.title} › ${source.section}`,
        providerMetadata: { pharma: source },
      })),
    ],
    metadata: {
      status: "completed",
      createdAt: CREATED_AT,
      feedback: null,
      ...metadata,
    },
  }
}

export function conversationTurns(count: number, prefix: string): ApiUIMessage[] {
  return Array.from({ length: count }, (_unused, position) => {
    const turn = position + 1
    return [
      apiUserMessage(`${prefix}-u${turn}`, `Câu hỏi ${prefix} ${turn}`),
      apiAssistantMessage(
        `${prefix}-a${turn}`,
        `Trả lời ${prefix} ${turn}\n\n${"Nội dung chi tiết. ".repeat(12)}`
      ),
    ]
  }).flat()
}

export function messagePage(
  items: ApiUIMessage[],
  nextCursor: string | null = null
): MessagePage {
  return { items, next_cursor: nextCursor }
}

export function conversationView(
  id: string,
  title: string,
  overrides: Partial<ConversationView> = {}
): ConversationView {
  return {
    id,
    title,
    turn_count: 1,
    created_at: CREATED_AT,
    updated_at: CREATED_AT,
    ...overrides,
  }
}

export function conversationPage(
  items: ConversationView[],
  nextCursor: string | null = null
): ConversationPage {
  return { items, next_cursor: nextCursor }
}

export function citationDetail(
  index: number,
  overrides: Partial<CitationDetail> = {}
): CitationDetail {
  return {
    index,
    source: "Dược thư Quốc gia Việt Nam (Xuất bản lần thứ 2)",
    document_title: "Paracetamol",
    section: "Liều lượng và cách dùng",
    start_page: 811,
    end_page: 812,
    strategy: "full_section",
    is_current: true,
    chunks: [
      {
        id: "7b0f4f4e-2c1f-4f0e-9a5e-000000000000",
        text: "**Liều thường dùng**",
        matched: false,
        start_page: 811,
        end_page: 811,
      },
      {
        id: "7b0f4f4e-2c1f-4f0e-9a5e-000000000001",
        text: "Người lớn và trẻ em trên 12 tuổi uống 0,5–1 g mỗi 4–6 giờ.",
        matched: true,
        start_page: 812,
        end_page: 812,
      },
    ],
    ...overrides,
  }
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd frontend && npx vitest run app/features/chat/test-support.browser.test.tsx app/i18n/resources.test.ts`
Expected: PASS (2 browser tests; P8's resource parity tests still pass).

- [ ] **Step 6: Full check**

Run: `cd frontend && npm run format:write && npm run lint && npm run format && npm run typecheck && npm test`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add frontend/app/i18n/resources/vi/chat.json frontend/app/i18n/resources/en/chat.json \
  frontend/app/i18n/resources/vi/citations.json frontend/app/i18n/resources/en/citations.json \
  frontend/tests/chat/render.tsx frontend/tests/chat/fixtures.ts \
  frontend/app/features/chat/test-support.browser.test.tsx
git commit -m "feat(chat): add chat and citation copy and chat test support" \
  -m "$COMMIT_TRAILER"
```

---

### Task 2: Chat libraries, shadcn chat components and adapted elements

**Files:**
- Modify: `frontend/package.json`, `frontend/package-lock.json`, `frontend/app/app.css`
- Create (CLI): `frontend/app/components/ui/{message-scroller,message,bubble,marker,collapsible,alert,empty}.tsx`
- Create: `frontend/app/components/elements/{surfaces,agent-status,retrieval-chunks,sources,inline-citation,feedback-dialog,guardrail-notice,error-state}.tsx`
- Test: `frontend/app/components/elements/elements.browser.test.tsx`

**Interfaces:**
- Consumes: `renderWithProviders` (Task 1); shadcn `collapsible` (added here).
- Produces:
  - `surfaces.tsx`: class strings `paper`, `floating`, `field`, `fieldInteractive`, `inkButton`, `collapsePanel`, `mono`; `ShimmerLabel(props: ComponentProps<"span"> & { active?: boolean })`
  - `AgentStatus(props: Omit<ComponentProps<"div">, "children"> & { state: AgentState; label: string; elapsed?: string })`, `type AgentState = "working" | "waiting" | "done"`
  - `RetrievalChunks(props: { chunks: readonly RetrievalChunk[]; searching: boolean; open: boolean; onOpenChange: (open: boolean) => void; className?: string })`, `interface RetrievalChunk { id: string; source: string; locator: string; text: string }`
  - `Sources(props: { sources: readonly SourceCard[]; open: boolean; onOpenChange: (open: boolean) => void; onSelect: (index: number) => void; className?: string })`, `interface SourceCard { index: number; source: string; title: string; stale: boolean }`
  - `Citation(props: { index: number; preview: CitationPreview; open: boolean; onOpenChange: (open: boolean) => void; onSelect: () => void })` (a button whose accessible name is the index), `interface CitationPreview { source: string; heading: string; pages: string | null; snippet: string; note: string | null }`
  - `FeedbackDialog(props: { reasons: readonly FeedbackReason[]; selected: readonly string[]; note: string; sent: boolean; onToggleReason: (id: string) => void; onNoteChange: (note: string) => void; onSubmit: () => void; className?: string })`, `interface FeedbackReason { id: string; label: string }`
  - `GuardrailNotice(props: { title: string; children: ReactNode; className?: string })`
  - `ErrorState(props: { title: string; detail: string; retrying: boolean; onRetry: () => void; className?: string })`

The registry files are demos: hard-coded English, a demo paragraph in `InlineCitation`, relevance scores, a `SwapLabel` helper whose effect breaks `react/exhaustive-deps`, and a `range` util only the score bar needs. The adapted copies keep the look, take their labels from i18n and drop those parts.

- [ ] **Step 1: Write the failing test**

`frontend/app/components/elements/elements.browser.test.tsx`:

```tsx
import { useState } from "react"
import { describe, expect, test, vi } from "vitest"
import { page, userEvent } from "vitest/browser"

import { renderWithProviders } from "../../../tests/chat/render"

import { AgentStatus } from "./agent-status"
import { ErrorState } from "./error-state"
import { FeedbackDialog } from "./feedback-dialog"
import { GuardrailNotice } from "./guardrail-notice"
import { Citation } from "./inline-citation"
import { RetrievalChunks } from "./retrieval-chunks"
import { Sources } from "./sources"

describe("adapted elements", () => {
  test("AgentStatus shows the label and a localized screen reader state", async () => {
    const screen = await renderWithProviders(
      <AgentStatus state="working" label="Đang tìm kiếm · vòng 2" />
    )
    await expect.element(screen.getByText("Đang tìm kiếm · vòng 2")).toBeVisible()
    await expect.element(screen.getByRole("status")).toHaveTextContent("Đang xử lý")
  })

  test("ErrorState retries with a Vietnamese button", async () => {
    const onRetry = vi.fn<() => void>()
    const screen = await renderWithProviders(
      <ErrorState
        title="Chưa trả lời được"
        detail="Agent phản hồi quá lâu."
        retrying={false}
        onRetry={onRetry}
      />
    )
    await screen.getByRole("button", { name: "Thử lại" }).click()
    expect(onRetry).toHaveBeenCalledOnce()
  })

  test("GuardrailNotice renders its title and body", async () => {
    const screen = await renderWithProviders(
      <GuardrailNotice title="Không thể trả lời câu hỏi này">
        Nội dung giải thích
      </GuardrailNotice>
    )
    await expect.element(screen.getByText("Không thể trả lời câu hỏi này")).toBeVisible()
    await expect.element(screen.getByText("Nội dung giải thích")).toBeVisible()
  })

  test("RetrievalChunks shows the read count and no score", async () => {
    function Harness() {
      const [open, setOpen] = useState(true)
      return (
        <RetrievalChunks
          chunks={[
            { id: "1", source: "Paracetamol", locator: "Trang 812", text: "Người lớn 500 mg" },
          ]}
          searching={false}
          open={open}
          onOpenChange={setOpen}
        />
      )
    }
    const screen = await renderWithProviders(<Harness />)
    await expect.element(screen.getByText("Đã đọc 1 đoạn")).toBeVisible()
    await expect.element(screen.getByText("Người lớn 500 mg")).toBeVisible()
    expect(screen.container.querySelector("[role=meter]")).toBeNull()
  })

  test("Sources expands and selects a card by citation index", async () => {
    const onSelect = vi.fn<(index: number) => void>()
    function Harness() {
      const [open, setOpen] = useState(false)
      return (
        <Sources
          sources={[
            { index: 1, source: "Dược thư Quốc gia", title: "Paracetamol › Liều dùng", stale: false },
            { index: 2, source: "Dược thư Quốc gia", title: "Ibuprofen › Tương tác", stale: true },
          ]}
          open={open}
          onOpenChange={setOpen}
          onSelect={onSelect}
        />
      )
    }
    const screen = await renderWithProviders(<Harness />)
    await screen.getByRole("button", { name: "2 nguồn" }).click()
    await expect.element(screen.getByText("Nguồn đã có phiên bản mới hơn")).toBeVisible()
    await screen.getByRole("button", { name: /Ibuprofen › Tương tác/ }).click()
    expect(onSelect).toHaveBeenCalledWith(2)
  })

  test("Citation is named by its index, previews on hover and selects on click", async () => {
    const onSelect = vi.fn<() => void>()
    function Harness() {
      const [open, setOpen] = useState(false)
      return (
        <p>
          Liều 500 mg
          <Citation
            index={1}
            preview={{
              source: "Dược thư Quốc gia",
              heading: "Trích từ mục Paracetamol › Liều dùng",
              pages: "Trang 812–813",
              snippet: "Người lớn và trẻ em trên 12 tuổi…",
              note: null,
            }}
            open={open}
            onOpenChange={setOpen}
            onSelect={onSelect}
          />
        </p>
      )
    }
    const screen = await renderWithProviders(<Harness />)
    const trigger = screen.getByRole("button", { name: "1", exact: true })
    await userEvent.hover(trigger)
    await expect.element(page.getByText("Người lớn và trẻ em trên 12 tuổi…")).toBeVisible()
    await trigger.click()
    expect(onSelect).toHaveBeenCalledOnce()
  })

  test("FeedbackDialog toggles reasons, takes a note and submits", async () => {
    const onToggleReason = vi.fn<(id: string) => void>()
    const onNoteChange = vi.fn<(note: string) => void>()
    const onSubmit = vi.fn<() => void>()
    const screen = await renderWithProviders(
      <FeedbackDialog
        reasons={[{ id: "incorrect", label: "Sai thông tin" }]}
        selected={[]}
        note=""
        sent={false}
        onToggleReason={onToggleReason}
        onNoteChange={onNoteChange}
        onSubmit={onSubmit}
      />
    )
    await screen.getByRole("button", { name: "Sai thông tin" }).click()
    await userEvent.type(screen.getByRole("textbox", { name: "Ghi chú" }), "x")
    await screen.getByRole("button", { name: "Gửi phản hồi" }).click()
    expect(onToggleReason).toHaveBeenCalledWith("incorrect")
    expect(onNoteChange).toHaveBeenCalledWith("x")
    expect(onSubmit).toHaveBeenCalledOnce()
  })
})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd frontend && npx vitest run --project browser app/components/elements/elements.browser.test.tsx`
Expected: FAIL with `Failed to resolve import "./agent-status"`.

- [ ] **Step 3: Install the libraries and registry components**

```bash
cd frontend
npm install --save-exact ai@7.0.99 @ai-sdk/react@4.0.102 streamdown@2.6.0 @streamdown/code@1.1.1 @streamdown/math@1.0.2 react-intersection-observer@11.0.1
npm install --save-exact --save-dev @shadcn/helpers@0.2.0
npx shadcn@4.21.0 add message-scroller message bubble marker collapsible alert empty --yes
npx shadcn@4.21.0 add \
  https://r.assistant-ui.com/elements-agent-status.json \
  https://r.assistant-ui.com/elements-retrieval-chunks.json \
  https://r.assistant-ui.com/elements-sources.json \
  https://r.assistant-ui.com/elements-inline-citation.json \
  https://r.assistant-ui.com/elements-feedback-dialog.json \
  https://r.assistant-ui.com/elements-guardrail-notice.json \
  https://r.assistant-ui.com/elements-error-state.json --yes
npm install --save-exact @shadcn/react@0.3.1 tw-shimmer@0.4.13
npm install --save-exact "katex@$(node -p "require('katex/package.json').version")"
rm -rf app/components/assistant-ui
git status --short
```

Expected: `package.json` gains the dependencies with exact versions (`katex` at the version `@streamdown/math` installed, so the stylesheet matches the renderer); `app/components/ui/` gains the seven components (existing P8 files such as `button.tsx` are skipped); `app/app.css` gains `@import "tw-shimmer";`; the upstream element files the CLI wrote under `app/components/assistant-ui/` are removed because Step 5 writes the adapted copies. If the CLI re-ranged a P8 dependency, set it back to P8's exact version and run `npm install`.

- [ ] **Step 4: Add Streamdown styles and Tailwind sources to `app/app.css`**

Insert directly after the last `@import` line of `frontend/app/app.css`:

```css
@import "streamdown/styles.css";
@import "katex/dist/katex.min.css";

@source "../node_modules/streamdown/dist/*.js";
@source "../node_modules/@streamdown/code/dist/*.js";
@source "../node_modules/@streamdown/math/dist/*.js";
```

- [ ] **Step 5: Write the adapted elements**

`frontend/app/components/elements/surfaces.tsx`:

```tsx
import type { ComponentProps } from "react"

import { cn } from "~/lib/utils"

export const paper = "bg-background border border-border/60 dark:bg-popover"

export const floating = "bg-background border border-border/60 dark:bg-popover"

export const field = "bg-foreground/[0.04] dark:bg-foreground/[0.06]"

export const fieldInteractive =
  "bg-foreground/[0.04] transition-colors hover:bg-foreground/[0.07] dark:bg-foreground/[0.06] dark:hover:bg-foreground/[0.09]"

export const inkButton =
  "bg-foreground text-background transition-[opacity,scale] duration-150 ease-[cubic-bezier(0.23,1,0.32,1)] hover:opacity-90 active:scale-[0.96] motion-reduce:transition-none"

export const collapsePanel =
  "h-(--collapsible-panel-height) overflow-hidden transition-[height] duration-200 ease-[cubic-bezier(0.32,0.72,0,1)] data-[ending-style]:h-0 data-[starting-style]:h-0 motion-reduce:transition-none"

export const mono = "font-mono text-[11px] tracking-tight"

export function ShimmerLabel({
  active = true,
  className,
  ...props
}: ComponentProps<"span"> & { active?: boolean }) {
  return (
    <span
      className={cn(active && "shimmer motion-reduce:animate-none", className)}
      {...props}
    />
  )
}
```

`frontend/app/components/elements/agent-status.tsx`:

```tsx
import { CheckIcon } from "lucide-react"
import type { ComponentProps } from "react"
import { useTranslation } from "react-i18next"

import { cn } from "~/lib/utils"

import { mono, paper } from "./surfaces"

export type AgentState = "working" | "waiting" | "done"

export type AgentStatusProps = Omit<ComponentProps<"div">, "children"> & {
  state: AgentState
  label: string
  elapsed?: string
}

export function AgentStatus({
  state,
  label,
  elapsed,
  className,
  ...props
}: AgentStatusProps) {
  const { t } = useTranslation("chat")
  return (
    <div
      data-slot="agent-status"
      role="status"
      className={cn(
        paper,
        "flex w-fit items-center gap-2.5 rounded-full px-3.5 py-1.5",
        className
      )}
      {...props}
    >
      {state === "done" ? (
        <CheckIcon aria-hidden className="size-3 shrink-0 text-emerald-500" />
      ) : (
        <span
          aria-hidden
          className={cn(
            "size-1.5 shrink-0 rounded-full motion-reduce:animate-none",
            state === "working"
              ? "animate-pulse bg-blue-500 dark:bg-blue-400"
              : "border border-foreground/35"
          )}
        />
      )}
      <span className="sr-only">
        {state === "done" ? t("status.done") : t("status.working")}
      </span>
      <span
        key={label}
        className="max-w-72 truncate text-xs duration-300 fade-in blur-in-[2px] animate-in motion-reduce:animate-none"
      >
        {label}
      </span>
      {elapsed !== undefined && state !== "done" ? (
        <span className={cn(mono, "text-foreground/30 tabular-nums")}>
          {elapsed}
        </span>
      ) : null}
    </div>
  )
}
```

`frontend/app/components/elements/retrieval-chunks.tsx`:

```tsx
import { ChevronDownIcon, DatabaseIcon } from "lucide-react"
import { useTranslation } from "react-i18next"

import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "~/components/ui/collapsible"
import { cn } from "~/lib/utils"

import {
  collapsePanel,
  fieldInteractive,
  mono,
  paper,
  ShimmerLabel,
} from "./surfaces"

export interface RetrievalChunk {
  id: string
  source: string
  locator: string
  text: string
}

export type RetrievalChunksProps = {
  chunks: readonly RetrievalChunk[]
  searching: boolean
  open: boolean
  onOpenChange: (open: boolean) => void
  className?: string
}

export function RetrievalChunks({
  chunks,
  searching,
  open,
  onOpenChange,
  className,
}: RetrievalChunksProps) {
  const { t } = useTranslation("chat")
  return (
    <Collapsible
      data-slot="retrieval-chunks"
      open={open}
      onOpenChange={onOpenChange}
      className={cn("flex w-full max-w-xl flex-col", className)}
    >
      <CollapsibleTrigger
        className={cn(
          fieldInteractive,
          "group/trigger inline-flex w-fit items-center gap-1.5 rounded-full px-3.5 py-2 text-xs text-foreground/70 outline-none"
        )}
      >
        <DatabaseIcon aria-hidden className="size-3 text-foreground/40" />
        {searching ? (
          <ShimmerLabel className="relative inline-block leading-none">
            {t("evidence.searching")}
          </ShimmerLabel>
        ) : (
          <span>{t("evidence.read", { count: chunks.length })}</span>
        )}
        <ChevronDownIcon
          aria-hidden
          className="size-3 opacity-60 transition-transform duration-200 group-data-panel-open/trigger:rotate-180 motion-reduce:transition-none"
        />
      </CollapsibleTrigger>
      <CollapsibleContent className={cn(collapsePanel, "outline-none")}>
        <div className="flex flex-col gap-1.5 pt-2">
          {chunks.map((chunk) => (
            <div
              key={chunk.id}
              className={cn(
                paper,
                "flex flex-col gap-1.5 rounded-2xl px-3.5 py-2.5 duration-300 fade-in slide-in-from-bottom-1 animate-in fill-mode-both"
              )}
            >
              <div className="flex items-baseline gap-2">
                <span className="min-w-0 flex-1 truncate text-[13px] font-medium text-foreground/90">
                  {chunk.source}
                </span>
                <span className={cn(mono, "shrink-0 text-foreground/30")}>
                  {chunk.locator}
                </span>
              </div>
              <p className="line-clamp-2 text-xs leading-relaxed text-foreground/55">
                {chunk.text}
              </p>
            </div>
          ))}
        </div>
      </CollapsibleContent>
    </Collapsible>
  )
}
```

`frontend/app/components/elements/sources.tsx`:

```tsx
import { ChevronDownIcon } from "lucide-react"
import { useTranslation } from "react-i18next"

import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "~/components/ui/collapsible"
import { cn } from "~/lib/utils"

import { collapsePanel, fieldInteractive, mono, paper } from "./surfaces"

export interface SourceCard {
  index: number
  source: string
  title: string
  stale: boolean
}

export type SourcesProps = {
  sources: readonly SourceCard[]
  open: boolean
  onOpenChange: (open: boolean) => void
  onSelect: (index: number) => void
  className?: string
}

export function Sources({
  sources,
  open,
  onOpenChange,
  onSelect,
  className,
}: SourcesProps) {
  const { t } = useTranslation("chat")
  return (
    <Collapsible
      data-slot="sources"
      open={open}
      onOpenChange={onOpenChange}
      className={cn("w-full max-w-xl", className)}
    >
      <CollapsibleTrigger
        className={cn(
          fieldInteractive,
          "group/trigger inline-flex w-fit items-center gap-1.5 rounded-full px-3.5 py-2 text-xs text-foreground/60 outline-none hover:text-foreground/90"
        )}
      >
        <span>{t("sources.trigger", { count: sources.length })}</span>
        <ChevronDownIcon
          aria-hidden
          className="size-3 opacity-60 transition-transform duration-200 group-data-panel-open/trigger:rotate-180 motion-reduce:transition-none"
        />
      </CollapsibleTrigger>
      <CollapsibleContent className={cn(collapsePanel, "outline-none")}>
        <div className="grid grid-cols-1 gap-2 pt-2.5 sm:grid-cols-2">
          {sources.map((source) => (
            <button
              key={source.index}
              type="button"
              onClick={() => onSelect(source.index)}
              className={cn(
                paper,
                "flex flex-col gap-1.5 rounded-2xl p-3 text-start transition-transform hover:-translate-y-px"
              )}
            >
              <span className="flex items-center gap-1.5">
                <span className="flex h-4 min-w-4 shrink-0 items-center justify-center rounded bg-foreground/[0.06] px-1 text-[9px] font-medium text-foreground/45 tabular-nums">
                  {source.index}
                </span>
                <span className={cn(mono, "truncate text-foreground/40")}>
                  {source.source}
                </span>
              </span>
              <span className="line-clamp-2 text-[13px] leading-snug font-medium text-foreground/90">
                {source.title}
              </span>
              {source.stale ? (
                <span className="text-xs text-amber-600 dark:text-amber-400">
                  {t("sources.stale")}
                </span>
              ) : null}
            </button>
          ))}
        </div>
      </CollapsibleContent>
    </Collapsible>
  )
}
```

`frontend/app/components/elements/inline-citation.tsx` (the upstream `InlineCitation` demo paragraph is removed; the internal `Citation` is exported and takes localized strings):

```tsx
import { PreviewCard } from "@base-ui/react/preview-card"

import { cn } from "~/lib/utils"

import { floating, mono } from "./surfaces"

export interface CitationPreview {
  source: string
  heading: string
  pages: string | null
  snippet: string
  note: string | null
}

export type CitationProps = {
  index: number
  preview: CitationPreview
  open: boolean
  onOpenChange: (open: boolean) => void
  onSelect: () => void
}

export function Citation({
  index,
  preview,
  open,
  onOpenChange,
  onSelect,
}: CitationProps) {
  return (
    <PreviewCard.Root open={open} onOpenChange={onOpenChange}>
      <PreviewCard.Trigger
        delay={0}
        render={<button type="button" />}
        onClick={onSelect}
        className={cn(
          "mx-0.5 inline-flex h-4 min-w-4 translate-y-[-2px] cursor-pointer items-center justify-center rounded-[5px] px-1 align-middle font-mono text-[10px] font-medium tabular-nums transition-colors",
          open
            ? "bg-foreground text-background"
            : "bg-foreground/[0.06] text-foreground/45 hover:text-foreground/90"
        )}
      >
        {index}
      </PreviewCard.Trigger>
      <PreviewCard.Portal>
        <PreviewCard.Positioner side="top" sideOffset={8}>
          <PreviewCard.Popup
            className={cn(
              floating,
              "z-50 w-72 origin-(--transform-origin) rounded-2xl p-3.5 outline-none",
              "transition-[opacity,scale] duration-200 ease-[cubic-bezier(0.23,1,0.32,1)] motion-reduce:transition-none",
              "data-[starting-style]:scale-[0.97] data-[starting-style]:opacity-0",
              "data-[ending-style]:scale-[0.97] data-[ending-style]:opacity-0"
            )}
          >
            <div className="flex items-center gap-1.5">
              <span
                aria-hidden
                className="flex h-4 min-w-4 items-center justify-center rounded bg-foreground/[0.06] px-1 text-[9px] font-medium text-foreground/45 tabular-nums"
              >
                {index}
              </span>
              <span className={cn(mono, "line-clamp-1 text-foreground/40")}>
                {preview.source}
              </span>
            </div>
            <p className="mt-2 text-[13px] leading-snug font-medium">
              {preview.heading}
            </p>
            {preview.pages === null ? null : (
              <p className={cn(mono, "mt-0.5 text-foreground/40")}>
                {preview.pages}
              </p>
            )}
            <p className="mt-1 text-[13px] leading-relaxed text-foreground/50">
              {preview.snippet}
            </p>
            {preview.note === null ? null : (
              <p className="mt-1.5 text-xs text-amber-600 dark:text-amber-400">
                {preview.note}
              </p>
            )}
          </PreviewCard.Popup>
        </PreviewCard.Positioner>
      </PreviewCard.Portal>
    </PreviewCard.Root>
  )
}
```

`frontend/app/components/elements/feedback-dialog.tsx`:

```tsx
import { CheckIcon, ThumbsDownIcon } from "lucide-react"
import { useTranslation } from "react-i18next"

import { cn } from "~/lib/utils"

import { field, inkButton, mono } from "./surfaces"

export interface FeedbackReason {
  id: string
  label: string
}

export type FeedbackDialogProps = {
  reasons: readonly FeedbackReason[]
  selected: readonly string[]
  note: string
  sent: boolean
  onToggleReason: (id: string) => void
  onNoteChange: (note: string) => void
  onSubmit: () => void
  className?: string
}

export function FeedbackDialog({
  reasons,
  selected,
  note,
  sent,
  onToggleReason,
  onNoteChange,
  onSubmit,
  className,
}: FeedbackDialogProps) {
  const { t } = useTranslation("chat")
  return (
    <div
      data-slot="feedback-dialog"
      className={cn(
        "flex w-full",
        sent ? "items-center gap-2.5 text-[13.5px]" : "flex-col gap-3",
        className
      )}
    >
      <div
        role="status"
        className={
          sent ? "flex items-center gap-2.5 duration-300 fade-in animate-in" : "sr-only"
        }
      >
        {sent ? (
          <>
            <CheckIcon aria-hidden className="size-4 shrink-0 text-emerald-500" />
            {t("feedback.sent")}
          </>
        ) : null}
      </div>

      {sent ? null : (
        <>
          <div className="flex items-center gap-2.5">
            <span className="flex size-7 shrink-0 items-center justify-center rounded-lg bg-foreground/[0.05] text-foreground/45">
              <ThumbsDownIcon aria-hidden className="size-3.5" />
            </span>
            <span className="text-[13.5px] font-medium">{t("feedback.title")}</span>
            <span className={cn(mono, "ms-auto text-foreground/30")}>
              {t("feedback.optional")}
            </span>
          </div>

          <div className="flex flex-wrap gap-1.5">
            {reasons.map((reason) => {
              const active = selected.includes(reason.id)
              return (
                <button
                  key={reason.id}
                  type="button"
                  aria-pressed={active}
                  onClick={() => onToggleReason(reason.id)}
                  className={cn(
                    "rounded-full px-2.5 py-1 text-xs transition-[background-color,color,scale] duration-150 active:scale-[0.96]",
                    active
                      ? "bg-foreground text-background"
                      : cn(field, "text-foreground/55 hover:text-foreground/90")
                  )}
                >
                  {reason.label}
                </button>
              )
            })}
          </div>

          <textarea
            value={note}
            onChange={(event) => onNoteChange(event.target.value)}
            rows={3}
            placeholder={t("feedback.notePlaceholder")}
            aria-label={t("feedback.notePlaceholder")}
            className={cn(
              field,
              "resize-none rounded-xl px-3 py-2 text-xs text-foreground/80 outline-none placeholder:text-foreground/30 focus-visible:ring-1 focus-visible:ring-foreground/20"
            )}
          />

          <button
            type="button"
            onClick={onSubmit}
            className={cn(
              inkButton,
              "flex h-8 items-center justify-center self-end rounded-full px-3.5 text-xs font-medium"
            )}
          >
            {t("feedback.submit")}
          </button>
        </>
      )}
    </div>
  )
}
```

`frontend/app/components/elements/guardrail-notice.tsx`:

```tsx
import { ShieldIcon } from "lucide-react"
import type { ReactNode } from "react"

import { cn } from "~/lib/utils"

import { paper } from "./surfaces"

export type GuardrailNoticeProps = {
  title: string
  children: ReactNode
  className?: string
}

export function GuardrailNotice({ title, children, className }: GuardrailNoticeProps) {
  return (
    <div
      data-slot="guardrail-notice"
      className={cn(
        paper,
        "flex w-full max-w-xl flex-col gap-3 rounded-[20px] p-4",
        className
      )}
    >
      <div className="flex items-center gap-2.5">
        <span className="flex size-7 shrink-0 items-center justify-center rounded-lg bg-amber-500/12 text-amber-600 dark:text-amber-400">
          <ShieldIcon aria-hidden className="size-3.5" />
        </span>
        <span className="min-w-0 flex-1 text-[13.5px] font-medium">{title}</span>
      </div>
      <div className="text-sm leading-relaxed text-foreground/70">{children}</div>
    </div>
  )
}
```

`frontend/app/components/elements/error-state.tsx`:

```tsx
import { CircleAlertIcon, RefreshCwIcon } from "lucide-react"
import { useTranslation } from "react-i18next"

import { cn } from "~/lib/utils"

import { ShimmerLabel } from "./surfaces"

export type ErrorStateProps = {
  title: string
  detail: string
  retrying: boolean
  onRetry: () => void
  className?: string
}

export function ErrorState({
  title,
  detail,
  retrying,
  onRetry,
  className,
}: ErrorStateProps) {
  const { t } = useTranslation("chat")
  if (retrying) {
    return (
      <div
        data-slot="error-state"
        role="status"
        className={cn(
          "flex w-full max-w-xl items-center gap-2.5 text-sm duration-300 fade-in animate-in motion-reduce:animate-none",
          className
        )}
      >
        <RefreshCwIcon
          aria-hidden
          className="size-3.5 shrink-0 animate-spin text-foreground/45 motion-reduce:animate-none"
        />
        <ShimmerLabel className="relative inline-block text-foreground/55">
          {t("error.retrying")}
        </ShimmerLabel>
      </div>
    )
  }

  return (
    <div
      data-slot="error-state"
      role="alert"
      className={cn(
        "flex w-full max-w-xl items-start gap-2.5 rounded-2xl bg-red-500/[0.06] px-4 py-3 text-sm duration-300 fade-in animate-in motion-reduce:animate-none dark:bg-red-500/10",
        className
      )}
    >
      <CircleAlertIcon aria-hidden className="mt-0.5 size-4 shrink-0 text-red-500/80" />
      <div>
        <p className="font-medium text-red-600 dark:text-red-400">{title}</p>
        <p className="mt-0.5 text-[13px] leading-snug text-red-600/60 dark:text-red-400/60">
          {detail}
        </p>
      </div>
      <button
        type="button"
        onClick={onRetry}
        className="ms-auto flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium text-red-600 transition-colors hover:bg-red-500/10 dark:text-red-400"
      >
        <RefreshCwIcon aria-hidden className="size-3" />
        {t("error.retry")}
      </button>
    </div>
  )
}
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `cd frontend && npx vitest run --project browser app/components/elements/elements.browser.test.tsx`
Expected: PASS (7 tests).

- [ ] **Step 7: Full check**

Run: `cd frontend && npm run format:write && npm run lint && npm run format && npm run typecheck && npm test`
Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/app/app.css \
  frontend/app/components/ui/message-scroller.tsx frontend/app/components/ui/message.tsx \
  frontend/app/components/ui/bubble.tsx frontend/app/components/ui/marker.tsx \
  frontend/app/components/ui/collapsible.tsx frontend/app/components/ui/alert.tsx \
  frontend/app/components/ui/empty.tsx frontend/app/components/elements
git commit -m "feat(chat): add chat libraries, shadcn chat components and adapted elements" \
  -m "$COMMIT_TRAILER"
```

---

### Task 3: `PharmaUIMessage` types, stream schemas and history helpers

**Files:**
- Create: `frontend/app/features/chat/lib/message-schema.ts`, `frontend/app/features/chat/lib/chat-messages.ts`
- Test: `frontend/app/features/chat/lib/message-schema.test.ts`, `frontend/app/features/chat/lib/chat-messages.test.ts`

**Interfaces:**
- Consumes (orval, P6 models): types `ConversationData`, `EvidenceData`, `MessageFeedback`, `MessageMetadata`, `MessagePage`, `PharmaDataParts`, `PharmaSourceMetadata`, `PhaseData`, `SkillsData`, `UIMessage` (imported as `ApiUIMessage`) from `~/api/gen/schemas`; zod `ListMessagesResponse` from `~/api/gen/zod`; fixtures (Task 1).
- Produces (`message-schema.ts`): `AGENT_PHASES`; `type AgentPhase`; zod `pharmaDataPartSchemas` (`phase`, `skills`, `evidence`, `conversation`), `pharmaSourceMetadataSchema`, `messageMetadataSchema`; `type PharmaDataPartTypes = { [Name in keyof PharmaDataParts]: PharmaDataParts[Name] }`; `type PharmaUIMessage = UIMessage<MessageMetadata, PharmaDataPartTypes>`; `type PharmaMessagePart = PharmaUIMessage["parts"][number]`.
- Produces (`chat-messages.ts`): `toPharmaMessage(message: ApiUIMessage): PharmaUIMessage`; `prependOlder(current: readonly PharmaUIMessage[], older: readonly PharmaUIMessage[]): PharmaUIMessage[]`; `olderMessagesFrom(pages: readonly MessagePage[], appliedPageCount: number): PharmaUIMessage[]`; `textOf(message: PharmaUIMessage): string`; `questionBefore(messages: readonly PharmaUIMessage[], assistantMessageId: string): string | undefined`; `withFeedback(messages: readonly PharmaUIMessage[], messageId: string, feedback: MessageFeedback | null): PharmaUIMessage[]`; `dropTrailingTurn(messages: readonly PharmaUIMessage[]): PharmaUIMessage[]`.

Types come from P6's OpenAPI models. `PharmaDataParts` is an interface, which TypeScript does not treat as assignable to AI SDK's `UIDataTypes = Record<string, unknown>`; the mapped type `PharmaDataPartTypes` copies it into an object type that is. The zod schemas for data parts and source metadata are written here because orval's zod client emits `zod.unknown()` for a `text/event-stream` body; a two-way `toExtend` check in the test fails `npm run typecheck` if they drift from the generated types. `messageMetadataSchema` is taken from the generated `ListMessagesResponse`.

- [ ] **Step 1: Write the failing tests**

`frontend/app/features/chat/lib/message-schema.test.ts`:

```ts
import { describe, expect, expectTypeOf, test } from "vitest"
import type { z } from "zod"

import type {
  ConversationData,
  EvidenceData,
  MessageMetadata,
  PharmaSourceMetadata,
  PhaseData,
  SkillsData,
} from "~/api/gen/schemas"

import {
  messageMetadataSchema,
  pharmaDataPartSchemas,
  pharmaSourceMetadataSchema,
} from "./message-schema"

describe("message schemas", () => {
  test("agree with the generated types in both directions", () => {
    type Phase = z.infer<typeof pharmaDataPartSchemas.phase>
    type Skills = z.infer<typeof pharmaDataPartSchemas.skills>
    type Evidence = z.infer<typeof pharmaDataPartSchemas.evidence>
    type Conversation = z.infer<typeof pharmaDataPartSchemas.conversation>
    type Source = z.infer<typeof pharmaSourceMetadataSchema>
    expectTypeOf<Phase>().toExtend<PhaseData>()
    expectTypeOf<PhaseData>().toExtend<Phase>()
    expectTypeOf<Skills>().toExtend<SkillsData>()
    expectTypeOf<SkillsData>().toExtend<Skills>()
    expectTypeOf<Evidence>().toExtend<EvidenceData>()
    expectTypeOf<EvidenceData>().toExtend<Evidence>()
    expectTypeOf<Conversation>().toExtend<ConversationData>()
    expectTypeOf<ConversationData>().toExtend<Conversation>()
    expectTypeOf<Source>().toExtend<PharmaSourceMetadata>()
    expectTypeOf<PharmaSourceMetadata>().toExtend<Source>()
    expectTypeOf<z.infer<typeof messageMetadataSchema>>().toExtend<MessageMetadata>()
  })

  test("accept the stream data parts of spec A §3.2", () => {
    expect(
      pharmaDataPartSchemas.phase.parse({ phase: "searching", round: 2 })
    ).toEqual({ phase: "searching", round: 2 })
    expect(pharmaDataPartSchemas.phase.parse({ phase: "answering" }).phase).toBe(
      "answering"
    )
    expect(
      pharmaDataPartSchemas.skills.parse({
        skills: [{ name: "drug-monograph", title: "Chuyên luận thuốc" }],
      }).skills
    ).toHaveLength(1)
    expect(
      pharmaDataPartSchemas.conversation.parse({ id: "c1", title: "Liều paracetamol" })
        .title
    ).toBe("Liều paracetamol")
    expect(
      pharmaDataPartSchemas.evidence.parse({
        items: [
          {
            index: 1,
            source: "Dược thư",
            title: "Paracetamol",
            section: "Liều dùng",
            startPage: null,
            endPage: null,
            snippet: "…",
          },
        ],
      }).items[0]?.startPage
    ).toBeNull()
  })

  test("reject an unknown phase", () => {
    expect(pharmaDataPartSchemas.phase.safeParse({ phase: "thinking" }).success).toBe(
      false
    )
  })

  test("parse finish metadata from the stream", () => {
    const metadata = messageMetadataSchema.parse({
      status: "completed",
      errorCode: null,
      usage: { llmCalls: 4, promptTokens: 900, completionTokens: 120, searchRounds: 1 },
      runId: "00000000000000000000000000000003",
      persisted: false,
      createdAt: "2026-09-13T08:00:00Z",
    })
    expect(metadata.persisted).toBe(false)
  })
})
```

`frontend/app/features/chat/lib/chat-messages.test.ts`:

```ts
import { describe, expect, test } from "vitest"

import {
  apiAssistantMessage,
  apiUserMessage,
  conversationTurns,
  messagePage,
  pharmaSource,
} from "../../../../tests/chat/fixtures"

import {
  dropTrailingTurn,
  olderMessagesFrom,
  prependOlder,
  questionBefore,
  textOf,
  toPharmaMessage,
  withFeedback,
} from "./chat-messages"

describe("toPharmaMessage", () => {
  test("keeps text, source documents and metadata", () => {
    const message = toPharmaMessage(
      apiAssistantMessage("a1", "Liều 500 mg [1].", [pharmaSource(1)])
    )
    expect(message.role).toBe("assistant")
    expect(message.parts[0]).toEqual({ type: "text", text: "Liều 500 mg [1]." })
    expect(message.parts[1]).toMatchObject({
      type: "source-document",
      sourceId: "chunk-1",
      providerMetadata: { pharma: { index: 1, isCurrent: true } },
    })
    expect(message.metadata?.status).toBe("completed")
  })
})

describe("history merge", () => {
  test("prepends older messages and drops ids already present", () => {
    const current = conversationTurns(2, "new").map(toPharmaMessage)
    const older = [
      ...conversationTurns(1, "old"),
      apiUserMessage("new-u1", "trùng"),
    ].map(toPharmaMessage)
    expect(prependOlder(current, older).map((message) => message.id)).toEqual([
      "old-u1",
      "old-a1",
      "new-u1",
      "new-a1",
      "new-u2",
      "new-a2",
    ])
  })

  test("returns pages not applied yet, oldest first", () => {
    const pages = [
      messagePage(conversationTurns(1, "p0"), "cursor-1"),
      messagePage(conversationTurns(1, "p1"), "cursor-2"),
      messagePage(conversationTurns(1, "p2")),
    ]
    expect(olderMessagesFrom(pages, 1).map((message) => message.id)).toEqual([
      "p2-u1",
      "p2-a1",
      "p1-u1",
      "p1-a1",
    ])
    expect(olderMessagesFrom(pages, 3)).toEqual([])
  })
})

describe("message helpers", () => {
  const messages = [
    apiUserMessage("u1", "Liều paracetamol?"),
    apiAssistantMessage("a1", "Người lớn 500 mg [1].", [pharmaSource(1)]),
  ].map(toPharmaMessage)

  test("joins text parts", () => {
    const [, assistant] = messages
    expect(assistant === undefined ? "" : textOf(assistant)).toBe(
      "Người lớn 500 mg [1]."
    )
  })

  test("finds the question before an assistant message", () => {
    expect(questionBefore(messages, "a1")).toBe("Liều paracetamol?")
    expect(questionBefore(messages, "missing")).toBeUndefined()
  })

  test("sets feedback on one message", () => {
    const updated = withFeedback(messages, "a1", {
      rating: "down",
      note: "Thiếu liều trẻ em",
    })
    expect(updated[1]?.metadata?.feedback).toEqual({
      rating: "down",
      note: "Thiếu liều trẻ em",
    })
    expect(updated[0]).toBe(messages[0])
  })

  test("drops the trailing user and assistant pair", () => {
    expect(dropTrailingTurn(messages)).toEqual([])
    expect(dropTrailingTurn(messages.slice(0, 1))).toEqual([])
    expect(dropTrailingTurn([])).toEqual([])
  })
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd frontend && npx vitest run --project unit app/features/chat/lib`
Expected: FAIL with `Failed to resolve import "./message-schema"` and `"./chat-messages"`.

- [ ] **Step 3: Write `message-schema.ts`**

`frontend/app/features/chat/lib/message-schema.ts`:

```ts
import type { ChatInit, UIMessage } from "ai"
import { z } from "zod"

import type { MessageMetadata, PharmaDataParts } from "~/api/gen/schemas"
import { ListMessagesResponse } from "~/api/gen/zod"

export const AGENT_PHASES = [
  "guarding",
  "understanding",
  "selecting_skills",
  "searching",
  "reading",
  "answering",
] as const

export type AgentPhase = (typeof AGENT_PHASES)[number]

const evidenceItemSchema = z.object({
  index: z.number().int(),
  source: z.string(),
  title: z.string(),
  section: z.string(),
  startPage: z.number().int().nullable(),
  endPage: z.number().int().nullable(),
  snippet: z.string(),
})

export const pharmaSourceMetadataSchema = evidenceItemSchema.extend({
  isCurrent: z.boolean(),
})

export type PharmaDataPartTypes = {
  [Name in keyof PharmaDataParts]: PharmaDataParts[Name]
}

export type PharmaUIMessage = UIMessage<MessageMetadata, PharmaDataPartTypes>

export type PharmaMessagePart = PharmaUIMessage["parts"][number]

export const pharmaDataPartSchemas = {
  phase: z.object({
    phase: z.enum(AGENT_PHASES),
    round: z.number().int().nullable().optional(),
  }),
  skills: z.object({
    skills: z.array(z.object({ name: z.string(), title: z.string() })),
  }),
  evidence: z.object({ items: z.array(evidenceItemSchema) }),
  conversation: z.object({ id: z.string(), title: z.string() }),
} satisfies NonNullable<ChatInit<PharmaUIMessage>["dataPartSchemas"]>

export const messageMetadataSchema =
  ListMessagesResponse.shape.items.element.shape.metadata satisfies NonNullable<
    ChatInit<PharmaUIMessage>["messageMetadataSchema"]
  >
```

- [ ] **Step 4: Write `chat-messages.ts`**

`frontend/app/features/chat/lib/chat-messages.ts`:

```ts
import type {
  UIMessage as ApiUIMessage,
  MessageFeedback,
  MessagePage,
} from "~/api/gen/schemas"

import type { PharmaMessagePart, PharmaUIMessage } from "./message-schema"

export function toPharmaMessage(message: ApiUIMessage): PharmaUIMessage {
  return {
    id: message.id,
    role: message.role,
    metadata: message.metadata,
    parts: message.parts.map((part): PharmaMessagePart => {
      if (part.type === "text") {
        return { type: "text", text: part.text }
      }
      return {
        type: "source-document",
        sourceId: part.sourceId,
        mediaType: part.mediaType,
        title: part.title,
        providerMetadata: { pharma: { ...part.providerMetadata.pharma } },
      }
    }),
  }
}

export function prependOlder(
  current: readonly PharmaUIMessage[],
  older: readonly PharmaUIMessage[]
): PharmaUIMessage[] {
  const known = new Set(current.map((message) => message.id))
  return [...older.filter((message) => !known.has(message.id)), ...current]
}

export function olderMessagesFrom(
  pages: readonly MessagePage[],
  appliedPageCount: number
): PharmaUIMessage[] {
  return pages
    .slice(appliedPageCount)
    .toReversed()
    .flatMap((page) => page.items.map(toPharmaMessage))
}

export function textOf(message: PharmaUIMessage): string {
  return message.parts
    .flatMap((part) => (part.type === "text" ? [part.text] : []))
    .join("")
}

export function questionBefore(
  messages: readonly PharmaUIMessage[],
  assistantMessageId: string
): string | undefined {
  const position = messages.findIndex((message) => message.id === assistantMessageId)
  for (let index = position - 1; index >= 0; index -= 1) {
    const message = messages[index]
    if (message?.role === "user") {
      return textOf(message)
    }
  }
  return undefined
}

export function withFeedback(
  messages: readonly PharmaUIMessage[],
  messageId: string,
  feedback: MessageFeedback | null
): PharmaUIMessage[] {
  return messages.map((message) =>
    message.id === messageId && message.metadata !== undefined
      ? { ...message, metadata: { ...message.metadata, feedback } }
      : message
  )
}

export function dropTrailingTurn(
  messages: readonly PharmaUIMessage[]
): PharmaUIMessage[] {
  const withoutAssistant =
    messages.at(-1)?.role === "assistant" ? messages.length - 1 : messages.length
  const end =
    messages[withoutAssistant - 1]?.role === "user"
      ? withoutAssistant - 1
      : withoutAssistant
  return messages.slice(0, end)
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd frontend && npx vitest run --project unit app/features/chat/lib && npm run typecheck`
Expected: PASS (4 schema tests, 7 helper tests); typecheck green, which proves the two-way type checks and the `satisfies` checks against `ChatInit`.

- [ ] **Step 6: Full check**

Run: `cd frontend && npm run format:write && npm run lint && npm run format && npm run typecheck && npm test`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add frontend/app/features/chat/lib/message-schema.ts frontend/app/features/chat/lib/message-schema.test.ts \
  frontend/app/features/chat/lib/chat-messages.ts frontend/app/features/chat/lib/chat-messages.test.ts
git commit -m "feat(chat): add PharmaUIMessage schemas and history helpers" \
  -m "$COMMIT_TRAILER"
```

---

### Task 4: `toCiteRefMarkup`, `CitationMarker` and `AnswerMarkdown`

**Files:**
- Create: `frontend/app/features/chat/lib/cite-markers.ts`, `frontend/app/features/chat/lib/sources.ts`, `frontend/app/features/chat/lib/streamdown-config.ts`, `frontend/app/features/citations/lib/citation-format.ts`, `frontend/app/features/citations/citation-context.tsx`, `frontend/app/features/citations/CitationMarker.tsx`, `frontend/app/features/chat/components/AnswerMarkdown.tsx`
- Test: `frontend/app/features/chat/lib/cite-markers.test.ts`, `frontend/app/features/chat/lib/sources.test.ts`, `frontend/app/features/citations/lib/citation-format.test.ts`, `frontend/app/features/chat/components/AnswerMarkdown.browser.test.tsx`

**Interfaces:**
- Consumes: `Citation` (Task 2); `pharmaSourceMetadataSchema`, `PharmaUIMessage`, `toPharmaMessage` (Task 3); `renderWithProviders`, fixtures (Task 1).
- Produces:
  - `toCiteRefMarkup(text: string): string` (pinned by the overview) and `parseCiteIndex(value: unknown): number | null`
  - `readPharmaSource(part: SourceDocumentUIPart): PharmaSourceMetadata | undefined`; `citationSourcesOf(message: PharmaUIMessage): Map<number, PharmaSourceMetadata>`
  - `streamdownPlugins: PluginConfig` (`{ code, math }`)
  - `type PageLabel = { key: "pages.single"; page: number } | { key: "pages.range"; start: number; end: number } | null`; `pageLabel(startPage: number | null, endPage: number | null): PageLabel`; `usePageText(): (startPage: number | null, endPage: number | null) => string | null`
  - `type MessageCitations = { messageId: string; sources: ReadonlyMap<number, PharmaSourceMetadata>; openCitation: (index: number) => void }`; `MessageCitationsProvider({ value, children })`; `useMessageCitations(): MessageCitations | null`
  - `CitationMarker(props: Record<string, unknown>): JSX.Element | null` (Streamdown component for `cite-ref`; Streamdown types custom tags as `ComponentType<Record<string, unknown> & ExtraProps>`)
  - `AnswerMarkdown({ text, streaming }: { text: string; streaming: boolean })`

Streamdown receives `[n]` converted to `<cite-ref index="n"></cite-ref>`, with `allowedTags={{ "cite-ref": ["index"] }}`, `literalTagContent={["cite-ref"]}` and `components={{ "cite-ref": CitationMarker }}`. No custom `rehypePlugins` are passed, because that disables the `allowedTags` merge. A marker is a button named by its index (P10 locates it that way); tapping it on a touch device opens `CitationSheet` directly.

- [ ] **Step 1: Write the failing unit tests**

`frontend/app/features/chat/lib/cite-markers.test.ts`:

```ts
import { describe, expect, test } from "vitest"

import { parseCiteIndex, toCiteRefMarkup } from "./cite-markers"

describe("toCiteRefMarkup", () => {
  test("converts every complete marker", () => {
    expect(toCiteRefMarkup("Người lớn 500 mg [1] và trẻ em [12].")).toBe(
      'Người lớn 500 mg <cite-ref index="1"></cite-ref> và trẻ em <cite-ref index="12"></cite-ref>.'
    )
  })

  test("converts adjacent markers", () => {
    expect(toCiteRefMarkup("[1][2]")).toBe(
      '<cite-ref index="1"></cite-ref><cite-ref index="2"></cite-ref>'
    )
  })

  test("leaves links, words and partial markers alone", () => {
    expect(toCiteRefMarkup("[1](https://example.com)")).toBe("[1](https://example.com)")
    expect(toCiteRefMarkup("[a] [1")).toBe("[a] [1")
    expect(toCiteRefMarkup("[1234]")).toBe("[1234]")
  })
})

describe("parseCiteIndex", () => {
  test("accepts a positive integer string only", () => {
    expect(parseCiteIndex("3")).toBe(3)
    expect(parseCiteIndex("0")).toBeNull()
    expect(parseCiteIndex("x")).toBeNull()
    expect(parseCiteIndex(3)).toBeNull()
    expect(parseCiteIndex(undefined)).toBeNull()
  })
})
```

`frontend/app/features/chat/lib/sources.test.ts`:

```ts
import { describe, expect, test } from "vitest"

import { apiAssistantMessage, pharmaSource } from "../../../../tests/chat/fixtures"

import { toPharmaMessage } from "./chat-messages"
import { citationSourcesOf, readPharmaSource } from "./sources"

describe("sources", () => {
  test("indexes source documents by citation index", () => {
    const message = toPharmaMessage(
      apiAssistantMessage("a1", "[1] [2]", [
        pharmaSource(1),
        pharmaSource(2, { isCurrent: false }),
      ])
    )
    const sources = citationSourcesOf(message)
    expect([...sources.keys()]).toEqual([1, 2])
    expect(sources.get(2)?.isCurrent).toBe(false)
  })

  test("ignores a source document without pharma metadata", () => {
    expect(
      readPharmaSource({
        type: "source-document",
        sourceId: "x",
        mediaType: "text/markdown",
        title: "x",
      })
    ).toBeUndefined()
  })
})
```

`frontend/app/features/citations/lib/citation-format.test.ts`:

```ts
import { describe, expect, test } from "vitest"

import { pageLabel } from "./citation-format"

describe("pageLabel", () => {
  test("formats single pages and ranges", () => {
    expect(pageLabel(812, 813)).toEqual({ key: "pages.range", start: 812, end: 813 })
    expect(pageLabel(812, 812)).toEqual({ key: "pages.single", page: 812 })
    expect(pageLabel(812, null)).toEqual({ key: "pages.single", page: 812 })
    expect(pageLabel(null, 813)).toEqual({ key: "pages.single", page: 813 })
    expect(pageLabel(null, null)).toBeNull()
  })
})
```

- [ ] **Step 2: Write the failing browser test**

`frontend/app/features/chat/components/AnswerMarkdown.browser.test.tsx`:

```tsx
import { type ReactNode, useState } from "react"
import { describe, expect, test, vi } from "vitest"
import { page, userEvent } from "vitest/browser"

import type { PharmaSourceMetadata } from "~/api/gen/schemas"
import { MessageCitationsProvider } from "~/features/citations/citation-context"

import { pharmaSource } from "../../../../tests/chat/fixtures"
import { renderWithProviders } from "../../../../tests/chat/render"

import { AnswerMarkdown } from "./AnswerMarkdown"

function withCitations(
  sources: readonly PharmaSourceMetadata[],
  openCitation: (index: number) => void,
  children: ReactNode
) {
  return (
    <MessageCitationsProvider
      value={{
        messageId: "a1",
        sources: new Map(sources.map((source) => [source.index, source])),
        openCitation,
      }}
    >
      {children}
    </MessageCitationsProvider>
  )
}

describe("AnswerMarkdown", () => {
  test("renders a marker named by its index with a preview, and opens the citation", async () => {
    const openCitation = vi.fn<(index: number) => void>()
    const screen = await renderWithProviders(
      withCitations(
        [pharmaSource(1)],
        openCitation,
        <AnswerMarkdown
          text={"Người lớn uống **0,5–1 g** mỗi 4–6 giờ [1]."}
          streaming={false}
        />
      )
    )
    await expect.element(screen.getByText("0,5–1 g")).toBeVisible()
    const marker = screen.getByRole("button", { name: "1", exact: true })
    await userEvent.hover(marker)
    await expect
      .element(page.getByText("Trích từ mục Paracetamol › Liều lượng và cách dùng"))
      .toBeVisible()
    await expect.element(page.getByText("Trang 812–813")).toBeVisible()
    await expect
      .element(page.getByText("Người lớn và trẻ em trên 12 tuổi uống 0,5–1 g mỗi 4–6 giờ."))
      .toBeVisible()
    await marker.click()
    expect(openCitation).toHaveBeenCalledWith(1)
  })

  test("shows the stale note for a source that is no longer current", async () => {
    const screen = await renderWithProviders(
      withCitations(
        [pharmaSource(1, { isCurrent: false })],
        vi.fn<(index: number) => void>(),
        <AnswerMarkdown text="Liều [1]." streaming={false} />
      )
    )
    await userEvent.hover(screen.getByRole("button", { name: "1", exact: true }))
    await expect.element(page.getByText("Nguồn đã có phiên bản mới hơn")).toBeVisible()
  })

  test("renders a marker only once it is complete while streaming", async () => {
    function Harness() {
      const [text, setText] = useState("Đang viết [1")
      return (
        <>
          <AnswerMarkdown text={text} streaming />
          <button type="button" onClick={() => setText("Đang viết [1] xong")}>
            tiếp
          </button>
        </>
      )
    }
    const screen = await renderWithProviders(
      withCitations([pharmaSource(1)], vi.fn<(index: number) => void>(), <Harness />)
    )
    await expect.element(screen.getByText(/Đang viết/)).toBeVisible()
    expect(screen.getByRole("button", { name: "1", exact: true }).elements()).toHaveLength(0)
    await screen.getByRole("button", { name: "tiếp" }).click()
    await expect
      .element(screen.getByRole("button", { name: "1", exact: true }))
      .toBeVisible()
  })

  test("keeps an unknown marker as plain text", async () => {
    const screen = await renderWithProviders(
      withCitations(
        [],
        vi.fn<(index: number) => void>(),
        <AnswerMarkdown text="Không có nguồn [9]." streaming={false} />
      )
    )
    await expect.element(screen.getByText("[9]")).toBeVisible()
    expect(screen.container.querySelector("button")).toBeNull()
  })
})
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cd frontend && npx vitest run app/features/chat/lib/cite-markers.test.ts app/features/chat/lib/sources.test.ts app/features/citations/lib/citation-format.test.ts app/features/chat/components/AnswerMarkdown.browser.test.tsx`
Expected: FAIL with `Failed to resolve import "./cite-markers"`, `"./sources"`, `"./citation-format"` and `"./AnswerMarkdown"`.

- [ ] **Step 4: Write the helpers**

`frontend/app/features/chat/lib/cite-markers.ts`:

```ts
const CITE_MARKER = /\[(\d{1,3})\](?!\()/g
const CITE_INDEX = /^[1-9]\d{0,2}$/

export function toCiteRefMarkup(text: string): string {
  return text.replaceAll(
    CITE_MARKER,
    (_marker, index: string) => `<cite-ref index="${index}"></cite-ref>`
  )
}

export function parseCiteIndex(value: unknown): number | null {
  return typeof value === "string" && CITE_INDEX.test(value) ? Number(value) : null
}
```

`frontend/app/features/chat/lib/sources.ts`:

```ts
import type { SourceDocumentUIPart } from "ai"

import type { PharmaSourceMetadata } from "~/api/gen/schemas"

import { type PharmaUIMessage, pharmaSourceMetadataSchema } from "./message-schema"

export function readPharmaSource(
  part: SourceDocumentUIPart
): PharmaSourceMetadata | undefined {
  const parsed = pharmaSourceMetadataSchema.safeParse(part.providerMetadata?.["pharma"])
  return parsed.success ? parsed.data : undefined
}

export function citationSourcesOf(
  message: PharmaUIMessage
): Map<number, PharmaSourceMetadata> {
  const sources = new Map<number, PharmaSourceMetadata>()
  for (const part of message.parts) {
    if (part.type === "source-document") {
      const source = readPharmaSource(part)
      if (source !== undefined) {
        sources.set(source.index, source)
      }
    }
  }
  return sources
}
```

`frontend/app/features/chat/lib/streamdown-config.ts`:

```ts
import { code } from "@streamdown/code"
import { math } from "@streamdown/math"
import type { PluginConfig } from "streamdown"

export const streamdownPlugins: PluginConfig = { code, math }
```

`frontend/app/features/citations/lib/citation-format.ts`:

```ts
import { useCallback } from "react"
import { useTranslation } from "react-i18next"

export type PageLabel =
  | { key: "pages.single"; page: number }
  | { key: "pages.range"; start: number; end: number }
  | null

export function pageLabel(startPage: number | null, endPage: number | null): PageLabel {
  if (startPage === null && endPage === null) {
    return null
  }
  if (startPage === null || endPage === null || startPage === endPage) {
    return { key: "pages.single", page: startPage ?? endPage ?? 0 }
  }
  return { key: "pages.range", start: startPage, end: endPage }
}

export function usePageText(): (
  startPage: number | null,
  endPage: number | null
) => string | null {
  const { t } = useTranslation("citations")
  return useCallback(
    (startPage: number | null, endPage: number | null) => {
      const label = pageLabel(startPage, endPage)
      if (label === null) {
        return null
      }
      return label.key === "pages.single"
        ? t("pages.single", { page: label.page })
        : t("pages.range", { start: label.start, end: label.end })
    },
    [t]
  )
}
```

- [ ] **Step 5: Write the citation context, marker and answer renderer**

`frontend/app/features/citations/citation-context.tsx`:

```tsx
import { createContext, type ReactNode, useContext } from "react"

import type { PharmaSourceMetadata } from "~/api/gen/schemas"

export type MessageCitations = {
  messageId: string
  sources: ReadonlyMap<number, PharmaSourceMetadata>
  openCitation: (index: number) => void
}

const MessageCitationsContext = createContext<MessageCitations | null>(null)

export function MessageCitationsProvider({
  value,
  children,
}: {
  value: MessageCitations
  children: ReactNode
}) {
  return <MessageCitationsContext value={value}>{children}</MessageCitationsContext>
}

export function useMessageCitations(): MessageCitations | null {
  return useContext(MessageCitationsContext)
}
```

`frontend/app/features/citations/CitationMarker.tsx`:

```tsx
import { useState } from "react"
import { useTranslation } from "react-i18next"

import { Citation } from "~/components/elements/inline-citation"
import { parseCiteIndex } from "~/features/chat/lib/cite-markers"

import { useMessageCitations } from "./citation-context"
import { usePageText } from "./lib/citation-format"

export function CitationMarker(props: Record<string, unknown>) {
  const citations = useMessageCitations()
  const { t } = useTranslation("chat")
  const pageText = usePageText()
  const [open, setOpen] = useState(false)
  const index = parseCiteIndex(props["index"])
  if (index === null) {
    return null
  }
  const source = citations?.sources.get(index)
  if (citations === null || source === undefined) {
    return <span className="text-muted-foreground">{`[${index}]`}</span>
  }
  return (
    <Citation
      index={index}
      open={open}
      onOpenChange={setOpen}
      onSelect={() => {
        setOpen(false)
        citations.openCitation(index)
      }}
      preview={{
        source: source.source,
        heading: t("citation.excerpt", { title: source.title, section: source.section }),
        pages: pageText(source.startPage, source.endPage),
        snippet: source.snippet,
        note: source.isCurrent ? null : t("sources.stale"),
      }}
    />
  )
}
```

`frontend/app/features/chat/components/AnswerMarkdown.tsx`:

```tsx
import { useMemo } from "react"
import { type Components, Streamdown } from "streamdown"

import { CitationMarker } from "~/features/citations/CitationMarker"

import { toCiteRefMarkup } from "../lib/cite-markers"
import { streamdownPlugins } from "../lib/streamdown-config"

const allowedTags = { "cite-ref": ["index"] }
const literalTagContent = ["cite-ref"]
const components: Components = { "cite-ref": CitationMarker }

export type AnswerMarkdownProps = {
  text: string
  streaming: boolean
}

export function AnswerMarkdown({ text, streaming }: AnswerMarkdownProps) {
  const markup = useMemo(() => toCiteRefMarkup(text), [text])
  return (
    <Streamdown
      className="text-sm leading-relaxed"
      mode={streaming ? "streaming" : "static"}
      isAnimating={streaming}
      plugins={streamdownPlugins}
      allowedTags={allowedTags}
      literalTagContent={literalTagContent}
      components={components}
    >
      {markup}
    </Streamdown>
  )
}
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cd frontend && npx vitest run app/features/chat/lib/cite-markers.test.ts app/features/chat/lib/sources.test.ts app/features/citations/lib/citation-format.test.ts app/features/chat/components/AnswerMarkdown.browser.test.tsx`
Expected: PASS (4 + 2 + 1 unit tests, 4 browser tests).

- [ ] **Step 7: Full check**

Run: `cd frontend && npm run format:write && npm run lint && npm run format && npm run typecheck && npm test`
Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add frontend/app/features/chat/lib/cite-markers.ts frontend/app/features/chat/lib/cite-markers.test.ts \
  frontend/app/features/chat/lib/sources.ts frontend/app/features/chat/lib/sources.test.ts \
  frontend/app/features/chat/lib/streamdown-config.ts \
  frontend/app/features/citations/lib/citation-format.ts frontend/app/features/citations/lib/citation-format.test.ts \
  frontend/app/features/citations/citation-context.tsx frontend/app/features/citations/CitationMarker.tsx \
  frontend/app/features/chat/components/AnswerMarkdown.tsx frontend/app/features/chat/components/AnswerMarkdown.browser.test.tsx
git commit -m "feat(chat): render answers with Streamdown and citation markers" \
  -m "$COMMIT_TRAILER"
```

---

### Task 5: `describeMessage` and `MessageParts` (spec B §10.5)

**Files:**
- Create: `frontend/app/features/chat/lib/message-view.ts`, `frontend/app/features/chat/components/MessageParts.tsx`, `frontend/app/features/chat/components/EvidencePanel.tsx`, `frontend/app/features/chat/components/SourcesList.tsx`
- Test: `frontend/app/features/chat/lib/message-view.test.ts`, `frontend/app/features/chat/components/MessageParts.browser.test.tsx`

**Interfaces:**
- Consumes: `AgentStatus`, `ErrorState`, `GuardrailNotice`, `RetrievalChunks`, `Sources` (Task 2); shadcn `Badge`, `Bubble*`, `Marker*`, `Message*`; `PharmaUIMessage`, `AgentPhase`, `textOf`, `toPharmaMessage` (Task 3); `AnswerMarkdown`, `citationSourcesOf`, `MessageCitationsProvider`, `usePageText` (Task 4).
- Produces:
  - `type MessageNotice = { kind: "guardrail"; reason: "blocked" | "redirected" } | { kind: "error"; reason: "timeout" | "failed" | "network" } | null`
  - `type MessageView = { text: string; hasText: boolean; phase: PhaseData | undefined; showStatus: boolean; skills: SkillsData["skills"]; evidence: EvidenceData["items"]; sources: PharmaSourceMetadata[]; notice: MessageNotice; feedback: MessageFeedback | null; canGiveFeedback: boolean }`
  - `describeMessage(message: PharmaUIMessage, state: { streaming: boolean; failed: boolean }): MessageView`
  - `PHASE_KEYS: Record<AgentPhase, "phase.guarding" | "phase.understanding" | "phase.selecting_skills" | "phase.searching" | "phase.reading" | "phase.answering">`
  - `MessageParts(props: MessagePartsProps)` with `type MessagePartsProps = { message: PharmaUIMessage; streaming: boolean; stopped: boolean; failed: boolean; retrying: boolean; onRetry: (messageId: string) => void; onOpenCitation: (messageId: string, index: number) => void; renderActions?: (message: PharmaUIMessage, view: MessageView) => ReactNode }`
  - `EvidencePanel({ items, searching }: { items: EvidenceData["items"]; searching: boolean })`
  - `SourcesList({ sources, onSelect }: { sources: readonly PharmaSourceMetadata[]; onSelect: (index: number) => void })`

Rendering table (spec B §10.5):

| Data | Rendered as | Rule |
| --- | --- | --- |
| last `data-phase` | `AgentStatus` | only while the message streams; `round` present → `Đang tìm kiếm · vòng n` |
| last `data-skills` | `Badge` list | always |
| last `data-evidence` | `EvidencePanel` → `RetrievalChunks` | open while streaming without text, collapsed afterwards unless the user toggles it; no score |
| `text` parts joined | `AnswerMarkdown` | `streaming` drives Streamdown mode and `isAnimating` |
| `source-document` parts | `SourcesList` → `Sources` | sorted by index; `isCurrent === false` shows the stale line |
| `metadata.status ∈ {blocked, redirected}` | `GuardrailNotice` wrapping the text | replaces the normal answer and the sources list |
| `metadata.status ∈ {error, timeout}` or `failed` | `ErrorState` with retry | text (if any) stays above it |
| `metadata.status` `completed`, `partial`, `abstained` | normal answer | |
| `stopped` | `Marker` "Đã dừng" | set by `ChatSession` when `onFinish` reports `isAbort` |
| actions | `renderActions` in `MessageFooter` | only when `canGiveFeedback` (assistant, not streaming, has metadata, `persisted !== false`, not an error notice) |

`metadata.persisted === false` is a toast raised by `ChatSession` (Task 9).

- [ ] **Step 1: Write the failing unit test**

`frontend/app/features/chat/lib/message-view.test.ts`:

```ts
import { describe, expect, test } from "vitest"

import {
  apiAssistantMessage,
  apiUserMessage,
  pharmaSource,
} from "../../../../tests/chat/fixtures"

import { toPharmaMessage } from "./chat-messages"
import type { PharmaUIMessage } from "./message-schema"
import { describeMessage } from "./message-view"

const IDLE = { streaming: false, failed: false }

function streamingAssistant(parts: PharmaUIMessage["parts"]): PharmaUIMessage {
  return { id: "a1", role: "assistant", parts }
}

describe("describeMessage", () => {
  test("maps a streaming message before text arrives", () => {
    const view = describeMessage(
      streamingAssistant([
        { type: "data-phase", id: "phase", data: { phase: "understanding" } },
        { type: "data-phase", id: "phase", data: { phase: "searching", round: 2 } },
        {
          type: "data-skills",
          data: { skills: [{ name: "drug-monograph", title: "Chuyên luận thuốc" }] },
        },
        {
          type: "data-evidence",
          id: "evidence",
          data: {
            items: [
              {
                index: 1,
                source: "Dược thư",
                title: "Paracetamol",
                section: "Liều dùng",
                startPage: 812,
                endPage: 813,
                snippet: "Người lớn…",
              },
            ],
          },
        },
      ]),
      { streaming: true, failed: false }
    )
    expect(view.showStatus).toBe(true)
    expect(view.phase).toEqual({ phase: "searching", round: 2 })
    expect(view.skills).toEqual([{ name: "drug-monograph", title: "Chuyên luận thuốc" }])
    expect(view.evidence).toHaveLength(1)
    expect(view.hasText).toBe(false)
    expect(view.notice).toBeNull()
    expect(view.canGiveFeedback).toBe(false)
  })

  test("maps a completed history message with sorted sources", () => {
    const message = toPharmaMessage(
      apiAssistantMessage("a1", "Liều [2] [1].", [pharmaSource(2), pharmaSource(1)], {
        feedback: { rating: "up", note: "" },
      })
    )
    const view = describeMessage(message, IDLE)
    expect(view.text).toBe("Liều [2] [1].")
    expect(view.sources.map((source) => source.index)).toEqual([1, 2])
    expect(view.showStatus).toBe(false)
    expect(view.feedback).toEqual({ rating: "up", note: "" })
    expect(view.canGiveFeedback).toBe(true)
  })

  test("maps guardrail and error statuses and keeps abstained as an answer", () => {
    const of = (status: "blocked" | "redirected" | "timeout" | "error" | "abstained") =>
      describeMessage(
        toPharmaMessage(apiAssistantMessage(`a-${status}`, "Nội dung.", [], { status })),
        IDLE
      )
    expect(of("blocked").notice).toEqual({ kind: "guardrail", reason: "blocked" })
    expect(of("redirected").notice).toEqual({ kind: "guardrail", reason: "redirected" })
    expect(of("timeout").notice).toEqual({ kind: "error", reason: "timeout" })
    expect(of("error").notice).toEqual({ kind: "error", reason: "failed" })
    expect(of("timeout").canGiveFeedback).toBe(false)
    expect(of("abstained").notice).toBeNull()
  })

  test("maps a transport failure without metadata to a network error", () => {
    const view = describeMessage(streamingAssistant([{ type: "text", text: "Đang" }]), {
      streaming: false,
      failed: true,
    })
    expect(view.notice).toEqual({ kind: "error", reason: "network" })
  })

  test("refuses feedback on a turn that was not persisted", () => {
    const message = toPharmaMessage(
      apiAssistantMessage("a1", "Liều.", [], { persisted: false })
    )
    expect(describeMessage(message, IDLE).canGiveFeedback).toBe(false)
  })

  test("maps a user message to its text", () => {
    const view = describeMessage(
      toPharmaMessage(apiUserMessage("u1", "Liều paracetamol?")),
      IDLE
    )
    expect(view.text).toBe("Liều paracetamol?")
    expect(view.canGiveFeedback).toBe(false)
  })
})
```

- [ ] **Step 2: Write the failing browser test**

`frontend/app/features/chat/components/MessageParts.browser.test.tsx`:

```tsx
import { describe, expect, test, vi } from "vitest"

import {
  apiAssistantMessage,
  apiUserMessage,
  pharmaSource,
} from "../../../../tests/chat/fixtures"
import { renderWithProviders } from "../../../../tests/chat/render"
import { toPharmaMessage } from "../lib/chat-messages"
import type { PharmaUIMessage } from "../lib/message-schema"

import { MessageParts, type MessagePartsProps } from "./MessageParts"

async function renderParts(
  overrides: Partial<MessagePartsProps> & { message: PharmaUIMessage }
) {
  const props: MessagePartsProps = {
    streaming: false,
    stopped: false,
    failed: false,
    retrying: false,
    onRetry: vi.fn<(messageId: string) => void>(),
    onOpenCitation: vi.fn<(messageId: string, index: number) => void>(),
    ...overrides,
  }
  const screen = await renderWithProviders(<MessageParts {...props} />)
  return { screen, props }
}

describe("MessageParts", () => {
  test("shows phase, skills and evidence while streaming", async () => {
    const { screen } = await renderParts({
      streaming: true,
      message: {
        id: "a1",
        role: "assistant",
        parts: [
          { type: "data-phase", id: "phase", data: { phase: "searching", round: 2 } },
          {
            type: "data-skills",
            data: { skills: [{ name: "drug-monograph", title: "Chuyên luận thuốc" }] },
          },
          {
            type: "data-evidence",
            id: "evidence",
            data: {
              items: [
                {
                  index: 1,
                  source: "Dược thư",
                  title: "Paracetamol",
                  section: "Liều dùng",
                  startPage: 812,
                  endPage: 812,
                  snippet: "Người lớn uống 500 mg",
                },
              ],
            },
          },
        ],
      },
    })
    await expect.element(screen.getByText("Đang tìm kiếm · vòng 2")).toBeVisible()
    await expect.element(screen.getByText("Chuyên luận thuốc")).toBeVisible()
    await expect.element(screen.getByText("Đang tìm tài liệu")).toBeVisible()
    await expect.element(screen.getByText("Người lớn uống 500 mg")).toBeVisible()
    await expect.element(screen.getByText("Trang 812")).toBeVisible()
  })

  test("renders a completed answer with sources that open citations", async () => {
    const { screen, props } = await renderParts({
      message: toPharmaMessage(
        apiAssistantMessage("a1", "Người lớn 500 mg [1], trẻ em theo cân nặng [2].", [
          pharmaSource(1),
          pharmaSource(2, { title: "Ibuprofen", section: "Tương tác", isCurrent: false }),
        ])
      ),
    })
    await expect.element(screen.getByText(/Người lớn 500 mg/)).toBeVisible()
    await screen.getByRole("button", { name: "2 nguồn" }).click()
    await screen.getByRole("button", { name: /Ibuprofen › Tương tác/ }).click()
    expect(props.onOpenCitation).toHaveBeenCalledWith("a1", 2)
    expect(screen.container.querySelector("[data-slot=agent-status]")).toBeNull()
  })

  test("replaces the answer with a guardrail notice", async () => {
    const { screen } = await renderParts({
      message: toPharmaMessage(
        apiAssistantMessage("a1", "Mình chỉ hỗ trợ tra cứu thuốc.", [], {
          status: "redirected",
        })
      ),
    })
    await expect
      .element(screen.getByText("Câu hỏi nằm ngoài phạm vi tra cứu thuốc"))
      .toBeVisible()
    await expect.element(screen.getByText("Mình chỉ hỗ trợ tra cứu thuốc.")).toBeVisible()
  })

  test("shows a timeout with retry", async () => {
    const { screen, props } = await renderParts({
      message: toPharmaMessage(apiAssistantMessage("a1", "", [], { status: "timeout" })),
    })
    await expect.element(screen.getByText("Agent phản hồi quá lâu.")).toBeVisible()
    await screen.getByRole("button", { name: "Thử lại" }).click()
    expect(props.onRetry).toHaveBeenCalledWith("a1")
  })

  test("marks a stopped message", async () => {
    const { screen } = await renderParts({
      stopped: true,
      message: { id: "a1", role: "assistant", parts: [{ type: "text", text: "Người lớn" }] },
    })
    await expect.element(screen.getByText("Đã dừng")).toBeVisible()
  })

  test("renders actions only for a message that accepts feedback", async () => {
    const { screen } = await renderParts({
      message: toPharmaMessage(apiAssistantMessage("a1", "Liều.")),
      renderActions: (message) => <span>{`actions-${message.id}`}</span>,
    })
    await expect.element(screen.getByText("actions-a1")).toBeVisible()
  })

  test("renders a user message in a bubble", async () => {
    const { screen } = await renderParts({
      message: toPharmaMessage(apiUserMessage("u1", "Liều paracetamol?")),
    })
    await expect.element(screen.getByText("Liều paracetamol?")).toBeVisible()
  })
})
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cd frontend && npx vitest run app/features/chat/lib/message-view.test.ts app/features/chat/components/MessageParts.browser.test.tsx`
Expected: FAIL with `Failed to resolve import "./message-view"` and `"./MessageParts"`.

- [ ] **Step 4: Write `message-view.ts`**

`frontend/app/features/chat/lib/message-view.ts`:

```ts
import type {
  EvidenceData,
  MessageFeedback,
  PharmaSourceMetadata,
  PhaseData,
  SkillsData,
} from "~/api/gen/schemas"

import { textOf } from "./chat-messages"
import type { AgentPhase, PharmaUIMessage } from "./message-schema"
import { citationSourcesOf } from "./sources"

export type MessageNotice =
  | { kind: "guardrail"; reason: "blocked" | "redirected" }
  | { kind: "error"; reason: "timeout" | "failed" | "network" }
  | null

export type MessageView = {
  text: string
  hasText: boolean
  phase: PhaseData | undefined
  showStatus: boolean
  skills: SkillsData["skills"]
  evidence: EvidenceData["items"]
  sources: PharmaSourceMetadata[]
  notice: MessageNotice
  feedback: MessageFeedback | null
  canGiveFeedback: boolean
}

export const PHASE_KEYS = {
  guarding: "phase.guarding",
  understanding: "phase.understanding",
  selecting_skills: "phase.selecting_skills",
  searching: "phase.searching",
  reading: "phase.reading",
  answering: "phase.answering",
} as const satisfies Record<AgentPhase, string>

function noticeOf(message: PharmaUIMessage, failed: boolean): MessageNotice {
  const status = message.metadata?.status
  if (status === "blocked" || status === "redirected") {
    return { kind: "guardrail", reason: status }
  }
  if (status === "timeout") {
    return { kind: "error", reason: "timeout" }
  }
  if (status === "error") {
    return { kind: "error", reason: "failed" }
  }
  return failed ? { kind: "error", reason: "network" } : null
}

export function describeMessage(
  message: PharmaUIMessage,
  { streaming, failed }: { streaming: boolean; failed: boolean }
): MessageView {
  let phase: PhaseData | undefined
  let skills: SkillsData["skills"] = []
  let evidence: EvidenceData["items"] = []
  for (const part of message.parts) {
    if (part.type === "data-phase") {
      phase = part.data
    } else if (part.type === "data-skills") {
      skills = part.data.skills
    } else if (part.type === "data-evidence") {
      evidence = part.data.items
    }
  }
  const text = textOf(message)
  const notice = message.role === "assistant" ? noticeOf(message, failed) : null
  const metadata = message.metadata
  return {
    text,
    hasText: text.trim().length > 0,
    phase,
    showStatus: streaming && phase !== undefined,
    skills,
    evidence,
    sources: [...citationSourcesOf(message).values()].toSorted(
      (left, right) => left.index - right.index
    ),
    notice,
    feedback: metadata?.feedback ?? null,
    canGiveFeedback:
      message.role === "assistant" &&
      !streaming &&
      metadata !== undefined &&
      metadata.persisted !== false &&
      notice?.kind !== "error",
  }
}
```

- [ ] **Step 5: Write the components**

`frontend/app/features/chat/components/EvidencePanel.tsx`:

```tsx
import { useMemo, useState } from "react"

import type { EvidenceData } from "~/api/gen/schemas"
import { RetrievalChunks } from "~/components/elements/retrieval-chunks"
import { usePageText } from "~/features/citations/lib/citation-format"

export type EvidencePanelProps = {
  items: EvidenceData["items"]
  searching: boolean
}

export function EvidencePanel({ items, searching }: EvidencePanelProps) {
  const pageText = usePageText()
  const [openOverride, setOpenOverride] = useState<boolean | null>(null)
  const chunks = useMemo(
    () =>
      items.map((item) => ({
        id: String(item.index),
        source: `${item.title} › ${item.section}`,
        locator: pageText(item.startPage, item.endPage) ?? "",
        text: item.snippet,
      })),
    [items, pageText]
  )
  return (
    <RetrievalChunks
      chunks={chunks}
      searching={searching}
      open={openOverride ?? searching}
      onOpenChange={setOpenOverride}
    />
  )
}
```

`frontend/app/features/chat/components/SourcesList.tsx`:

```tsx
import { useMemo, useState } from "react"

import type { PharmaSourceMetadata } from "~/api/gen/schemas"
import { Sources } from "~/components/elements/sources"

export type SourcesListProps = {
  sources: readonly PharmaSourceMetadata[]
  onSelect: (index: number) => void
}

export function SourcesList({ sources, onSelect }: SourcesListProps) {
  const [open, setOpen] = useState(false)
  const cards = useMemo(
    () =>
      sources.map((source) => ({
        index: source.index,
        source: source.source,
        title: `${source.title} › ${source.section}`,
        stale: !source.isCurrent,
      })),
    [sources]
  )
  return (
    <Sources sources={cards} open={open} onOpenChange={setOpen} onSelect={onSelect} />
  )
}
```

`frontend/app/features/chat/components/MessageParts.tsx`:

```tsx
import { type ReactNode, useMemo } from "react"
import { useTranslation } from "react-i18next"

import { AgentStatus } from "~/components/elements/agent-status"
import { ErrorState } from "~/components/elements/error-state"
import { GuardrailNotice } from "~/components/elements/guardrail-notice"
import { Badge } from "~/components/ui/badge"
import { Bubble, BubbleContent } from "~/components/ui/bubble"
import { Marker, MarkerContent } from "~/components/ui/marker"
import { Message, MessageContent, MessageFooter } from "~/components/ui/message"
import {
  type MessageCitations,
  MessageCitationsProvider,
} from "~/features/citations/citation-context"

import type { PharmaUIMessage } from "../lib/message-schema"
import { describeMessage, type MessageView, PHASE_KEYS } from "../lib/message-view"

import { AnswerMarkdown } from "./AnswerMarkdown"
import { EvidencePanel } from "./EvidencePanel"
import { SourcesList } from "./SourcesList"

export type MessagePartsProps = {
  message: PharmaUIMessage
  streaming: boolean
  stopped: boolean
  failed: boolean
  retrying: boolean
  onRetry: (messageId: string) => void
  onOpenCitation: (messageId: string, index: number) => void
  renderActions?: (message: PharmaUIMessage, view: MessageView) => ReactNode
}

export function MessageParts({
  message,
  streaming,
  stopped,
  failed,
  retrying,
  onRetry,
  onOpenCitation,
  renderActions,
}: MessagePartsProps) {
  const { t } = useTranslation("chat")
  const view = useMemo(
    () => describeMessage(message, { streaming, failed }),
    [message, streaming, failed]
  )
  const citations = useMemo<MessageCitations>(
    () => ({
      messageId: message.id,
      sources: new Map(view.sources.map((source) => [source.index, source])),
      openCitation: (index) => onOpenCitation(message.id, index),
    }),
    [message.id, view.sources, onOpenCitation]
  )

  if (message.role === "user") {
    return (
      <Message align="end">
        <MessageContent>
          <Bubble variant="secondary" align="end">
            <BubbleContent className="whitespace-pre-wrap">{view.text}</BubbleContent>
          </Bubble>
        </MessageContent>
      </Message>
    )
  }

  let phaseLabel = ""
  if (view.phase !== undefined) {
    phaseLabel =
      view.phase.phase === "searching" && typeof view.phase.round === "number"
        ? t("phase.searchingRound", { round: view.phase.round })
        : t(PHASE_KEYS[view.phase.phase])
  }

  let body: ReactNode = null
  if (view.notice?.kind === "guardrail") {
    body = (
      <GuardrailNotice
        title={
          view.notice.reason === "blocked"
            ? t("guardrail.blocked")
            : t("guardrail.redirected")
        }
      >
        <AnswerMarkdown text={view.text} streaming={false} />
      </GuardrailNotice>
    )
  } else if (view.notice?.kind === "error") {
    const detail =
      view.notice.reason === "timeout"
        ? t("error.timeout")
        : view.notice.reason === "network"
          ? t("error.network")
          : t("error.failed")
    body = (
      <>
        {view.hasText ? <AnswerMarkdown text={view.text} streaming={false} /> : null}
        <ErrorState
          title={t("error.title")}
          detail={detail}
          retrying={retrying}
          onRetry={() => onRetry(message.id)}
        />
      </>
    )
  } else if (view.hasText) {
    body = <AnswerMarkdown text={view.text} streaming={streaming} />
  }

  return (
    <Message align="start">
      <MessageContent>
        {view.showStatus ? <AgentStatus state="working" label={phaseLabel} /> : null}
        {view.skills.length > 0 ? (
          <ul
            aria-label={t("skills.label")}
            className="flex flex-wrap items-center gap-1.5"
          >
            {view.skills.map((skill) => (
              <li key={skill.name}>
                <Badge variant="secondary">{skill.title}</Badge>
              </li>
            ))}
          </ul>
        ) : null}
        {view.evidence.length > 0 ? (
          <EvidencePanel items={view.evidence} searching={streaming && !view.hasText} />
        ) : null}
        <MessageCitationsProvider value={citations}>{body}</MessageCitationsProvider>
        {view.sources.length > 0 && view.notice === null ? (
          <SourcesList sources={view.sources} onSelect={citations.openCitation} />
        ) : null}
        {stopped ? (
          <Marker variant="separator">
            <MarkerContent>{t("stopped")}</MarkerContent>
          </Marker>
        ) : null}
        {renderActions !== undefined && view.canGiveFeedback ? (
          <MessageFooter>{renderActions(message, view)}</MessageFooter>
        ) : null}
      </MessageContent>
    </Message>
  )
}
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cd frontend && npx vitest run app/features/chat/lib/message-view.test.ts app/features/chat/components/MessageParts.browser.test.tsx`
Expected: PASS (6 unit tests, 7 browser tests).

- [ ] **Step 7: Full check**

Run: `cd frontend && npm run format:write && npm run lint && npm run format && npm run typecheck && npm test`
Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add frontend/app/features/chat/lib/message-view.ts frontend/app/features/chat/lib/message-view.test.ts \
  frontend/app/features/chat/components/MessageParts.tsx frontend/app/features/chat/components/MessageParts.browser.test.tsx \
  frontend/app/features/chat/components/EvidencePanel.tsx frontend/app/features/chat/components/SourcesList.tsx
git commit -m "feat(chat): map message parts and metadata to the chat view" \
  -m "$COMMIT_TRAILER"
```

---

### Task 6: Chat transport and stream problem parsing

**Files:**
- Create: `frontend/app/features/chat/lib/chat-transport.ts`, `frontend/app/features/chat/lib/stream-problem.ts`
- Test: `frontend/app/features/chat/lib/stream-problem.test.ts`, `frontend/app/features/chat/lib/chat-transport.browser.test.tsx`

**Interfaces:**
- Consumes: `readCsrfToken`, `CSRF_HEADER` (`~/lib/csrf`); `ApiError`, `isApiError` (`~/api/problem`); `ChatRequest` (orval); `textOf`, `toPharmaMessage`, `PharmaUIMessage` (Task 3); `worker` (`tests/msw/browser.ts`); fixtures (Task 1).
- Produces:
  - `CHAT_STREAM_API = "/api/v1/chat/stream"`
  - `lastUserText(messages: readonly PharmaUIMessage[]): string`
  - `createChatTransport(): DefaultChatTransport<PharmaUIMessage>`
  - `readStreamProblem(error: Error): ApiError | null`

`DefaultChatTransport` in `ai` 7.0.99 throws `new Error(await response.text() || "Failed to fetch the chat response.")` for a non-OK response, so a problem+json body (401, 404, 422, 503 before the stream starts) reaches `useChat`'s `onError` as the error message. `readStreamProblem` parses it with zod into P8's `ApiError`, so the chat uses the same `isApiError` and `apiErrorMessage` as every other request. The transport resolves `headers` per request, so the CSRF cookie is read at send time.

- [ ] **Step 1: Write the failing tests**

`frontend/app/features/chat/lib/stream-problem.test.ts`:

```ts
import { describe, expect, test } from "vitest"

import { isApiError } from "~/api/problem"

import { readStreamProblem } from "./stream-problem"

describe("readStreamProblem", () => {
  test("turns a problem+json body carried in the error message into ApiError", () => {
    const problem = readStreamProblem(
      new Error(
        JSON.stringify({
          type: "urn:pharma-agent:problem:agent-unavailable",
          title: "Agent unavailable",
          status: 503,
          code: "AGENT_UNAVAILABLE",
        })
      )
    )
    expect(isApiError(problem)).toBe(true)
    expect(problem?.status).toBe(503)
    expect(problem?.code).toBe("AGENT_UNAVAILABLE")
    expect(problem?.message).toBe("Agent unavailable")
    expect(problem?.detail).toBeUndefined()
  })

  test("keeps the detail of a validation problem", () => {
    const problem = readStreamProblem(
      new Error(
        JSON.stringify({
          type: "urn:pharma-agent:problem:validation-error",
          title: "Validation error",
          status: 422,
          code: "VALIDATION_ERROR",
          detail: "Câu hỏi quá dài",
        })
      )
    )
    expect(problem?.detail).toBe("Câu hỏi quá dài")
  })

  test("returns null for a network error or JSON that is not a problem", () => {
    expect(readStreamProblem(new Error("Failed to fetch the chat response."))).toBeNull()
    expect(readStreamProblem(new TypeError("Failed to fetch"))).toBeNull()
    expect(readStreamProblem(new Error(JSON.stringify({ message: "x" })))).toBeNull()
  })
})
```

`frontend/app/features/chat/lib/chat-transport.browser.test.tsx`:

```tsx
import { createUIMessageStreamResponse, type UIMessageChunk } from "ai"
import { http, HttpResponse } from "msw"
import { afterEach, describe, expect, test } from "vitest"

import { apiAssistantMessage, apiUserMessage } from "../../../../tests/chat/fixtures"
import { worker } from "../../../../tests/msw/browser"

import { toPharmaMessage } from "./chat-messages"
import { CHAT_STREAM_API, createChatTransport } from "./chat-transport"
import { readStreamProblem } from "./stream-problem"

async function readAll<T>(stream: ReadableStream<T>): Promise<T[]> {
  const reader = stream.getReader()
  const values: T[] = []
  let result = await reader.read()
  while (!result.done) {
    values.push(result.value)
    result = await reader.read()
  }
  return values
}

const history = [
  apiUserMessage("u1", "Câu hỏi đầu"),
  apiAssistantMessage("a1", "Trả lời đầu"),
  apiUserMessage("u2", "Câu hỏi thứ hai"),
].map(toPharmaMessage)

function send() {
  return createChatTransport().sendMessages({
    trigger: "submit-message",
    chatId: "c1",
    messageId: undefined,
    messages: history,
    abortSignal: undefined,
  })
}

describe("createChatTransport", () => {
  afterEach(() => {
    document.cookie = "csrftoken=; max-age=0; path=/"
  })

  test("posts only the conversation id and the last question with the CSRF header", async () => {
    document.cookie = "csrftoken=token-123; path=/"
    const captured: { body: unknown; csrf: string | null }[] = []
    worker.use(
      http.post(`*${CHAT_STREAM_API}`, async ({ request }) => {
        captured.push({
          body: await request.json(),
          csrf: request.headers.get("x-csrftoken"),
        })
        const chunks: UIMessageChunk[] = [
          { type: "start", messageId: "a2" },
          { type: "finish", finishReason: "stop" },
        ]
        return createUIMessageStreamResponse({
          stream: new ReadableStream<UIMessageChunk>({
            start(controller) {
              for (const chunk of chunks) {
                controller.enqueue(chunk)
              }
              controller.close()
            },
          }),
        })
      })
    )

    const chunks = await readAll(await send())

    expect(captured).toEqual([
      {
        body: { conversation_id: "c1", message: "Câu hỏi thứ hai" },
        csrf: "token-123",
      },
    ])
    expect(chunks.map((chunk) => chunk.type)).toEqual(["start", "finish"])
  })

  test("rejects with the problem body when the stream cannot start", async () => {
    worker.use(
      http.post(`*${CHAT_STREAM_API}`, () =>
        HttpResponse.json(
          {
            type: "urn:pharma-agent:problem:agent-unavailable",
            title: "Agent unavailable",
            status: 503,
            code: "AGENT_UNAVAILABLE",
          },
          { status: 503, headers: { "Content-Type": "application/problem+json" } }
        )
      )
    )
    const error = await send().then(
      () => new Error("expected a rejection"),
      (reason: unknown) => (reason instanceof Error ? reason : new Error("not an Error"))
    )
    expect(readStreamProblem(error)?.code).toBe("AGENT_UNAVAILABLE")
  })
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd frontend && npx vitest run app/features/chat/lib/stream-problem.test.ts app/features/chat/lib/chat-transport.browser.test.tsx`
Expected: FAIL with `Failed to resolve import "./stream-problem"` and `"./chat-transport"`.

- [ ] **Step 3: Write `stream-problem.ts`**

`frontend/app/features/chat/lib/stream-problem.ts`:

```ts
import { z } from "zod"

import { ApiError } from "~/api/problem"

const streamProblemSchema = z.object({
  status: z.number().int(),
  code: z.string(),
  title: z.string().optional(),
  detail: z.string().nullish(),
})

export function readStreamProblem(error: Error): ApiError | null {
  let payload: unknown
  try {
    payload = JSON.parse(error.message)
  } catch {
    return null
  }
  const parsed = streamProblemSchema.safeParse(payload)
  if (!parsed.success) {
    return null
  }
  const { status, code, title, detail } = parsed.data
  return new ApiError({
    status,
    code,
    title: title ?? code,
    detail: detail ?? undefined,
  })
}
```

- [ ] **Step 4: Write `chat-transport.ts`**

`frontend/app/features/chat/lib/chat-transport.ts`:

```ts
import { DefaultChatTransport } from "ai"

import type { ChatRequest } from "~/api/gen/schemas"
import { CSRF_HEADER, readCsrfToken } from "~/lib/csrf"

import { textOf } from "./chat-messages"
import type { PharmaUIMessage } from "./message-schema"

export const CHAT_STREAM_API = "/api/v1/chat/stream"

function csrfHeaders(): Record<string, string> {
  const token = readCsrfToken()
  return token === undefined ? {} : { [CSRF_HEADER]: token }
}

export function lastUserText(messages: readonly PharmaUIMessage[]): string {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]
    if (message?.role === "user") {
      return textOf(message)
    }
  }
  return ""
}

export function createChatTransport(): DefaultChatTransport<PharmaUIMessage> {
  return new DefaultChatTransport<PharmaUIMessage>({
    api: CHAT_STREAM_API,
    credentials: "same-origin",
    headers: csrfHeaders,
    prepareSendMessagesRequest: ({ id, messages }) => {
      const body: ChatRequest = {
        conversation_id: id,
        message: lastUserText(messages),
      }
      return { body }
    },
  })
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd frontend && npx vitest run app/features/chat/lib/stream-problem.test.ts app/features/chat/lib/chat-transport.browser.test.tsx`
Expected: PASS (3 unit tests, 2 browser tests).

- [ ] **Step 6: Full check**

Run: `cd frontend && npm run format:write && npm run lint && npm run format && npm run typecheck && npm test`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add frontend/app/features/chat/lib/chat-transport.ts frontend/app/features/chat/lib/chat-transport.browser.test.tsx \
  frontend/app/features/chat/lib/stream-problem.ts frontend/app/features/chat/lib/stream-problem.test.ts
git commit -m "feat(chat): add the UI message stream transport and problem parsing" \
  -m "$COMMIT_TRAILER"
```

---

### Task 7: `CitationSheet` with the citation detail query (spec B §10.6)

**Files:**
- Create: `frontend/app/features/citations/CitationSheet.tsx`
- Test: `frontend/app/features/citations/CitationSheet.browser.test.tsx`

**Interfaces:**
- Consumes: `getGetMessageCitationQueryOptions(messageId, index)` (orval), `getGetMessageCitationMockHandler` (`~/api/gen/endpoints.msw`), `CitationDetail` (`~/api/gen/schemas`, snake_case, `strategy: HydrateStrategy`), `apiErrorMessage` (`~/i18n/error-message`), `useIsMobile` (`~/hooks/use-mobile`), shadcn `Sheet*`, `Drawer*`, `Badge`, `Spinner`; `usePageText`, `streamdownPlugins` (Task 4); `renderRoutes`, `citationDetail` (Task 1); `worker`.
- Produces: `type CitationTarget = { messageId: string; index: number }`; `CitationSheet({ target, onClose }: { target: CitationTarget | null; onClose: () => void })`; `CitationDetailView({ messageId, index }: CitationTarget)`.

The sheet stays mounted; `target !== null` opens it, and the last target stays rendered while the close animation runs. Desktop uses `Sheet` (a Base UI dialog on the right); below 768 px it uses `Drawer`. Chunks render in the order the backend returns (`block_chunk_version_ids`), each through Streamdown in static mode; the `matched` chunk is highlighted, labelled as a region and scrolled into view. A missing or foreign citation (404 `CITATION_NOT_FOUND`) shows the translated problem message.

- [ ] **Step 1: Write the failing test**

`frontend/app/features/citations/CitationSheet.browser.test.tsx`:

```tsx
import { http, HttpResponse } from "msw"
import { afterEach, describe, expect, test, vi } from "vitest"
import { page, userEvent } from "vitest/browser"

import { getGetMessageCitationMockHandler } from "~/api/gen/endpoints.msw"

import { citationDetail } from "../../../tests/chat/fixtures"
import { renderRoutes } from "../../../tests/chat/render"
import { worker } from "../../../tests/msw/browser"

import { CitationSheet } from "./CitationSheet"

function sheetPage(onClose: () => void) {
  return function Page() {
    return <CitationSheet target={{ messageId: "a1", index: 1 }} onClose={onClose} />
  }
}

describe("CitationSheet", () => {
  afterEach(async () => {
    await page.viewport(1280, 800)
  })

  test("shows the source header, strategy and the highlighted matched chunk", async () => {
    await page.viewport(1280, 800)
    worker.use(getGetMessageCitationMockHandler(citationDetail(1, { is_current: false })))
    await renderRoutes(sheetPage(vi.fn<() => void>()), { initialEntry: "/chat/c1" })

    await expect.element(page.getByRole("dialog")).toBeVisible()
    await expect.element(page.getByRole("heading", { name: "Nguồn [1]" })).toBeVisible()
    await expect
      .element(page.getByText("Paracetamol › Liều lượng và cách dùng"))
      .toBeVisible()
    await expect.element(page.getByText("Toàn bộ mục")).toBeVisible()
    await expect.element(page.getByText("Trang 811–812")).toBeVisible()
    await expect.element(page.getByText("Nguồn đã có phiên bản mới hơn")).toBeVisible()
    await expect.element(page.getByText("Liều thường dùng")).toBeVisible()
    const matched = page.getByRole("region", { name: "Đoạn được trích dẫn" })
    await expect.element(matched).toHaveTextContent("Người lớn và trẻ em trên 12 tuổi")
    await expect.element(matched).toHaveAttribute("data-matched", "true")
    expect(document.querySelector('[data-slot="sheet-content"]')).not.toBeNull()
  })

  test("shows the translated problem for a citation the user cannot read", async () => {
    worker.use(
      http.get("*/api/v1/messages/:messageId/citations/:index", () =>
        HttpResponse.json(
          {
            type: "urn:pharma-agent:problem:citation-not-found",
            title: "Citation not found",
            status: 404,
            code: "CITATION_NOT_FOUND",
          },
          { status: 404, headers: { "Content-Type": "application/problem+json" } }
        )
      )
    )
    await renderRoutes(sheetPage(vi.fn<() => void>()), { initialEntry: "/chat/c1" })
    await expect
      .element(page.getByRole("alert"))
      .toHaveTextContent("Không tìm thấy nguồn trích dẫn")
  })

  test("calls onClose when dismissed with Escape", async () => {
    worker.use(getGetMessageCitationMockHandler(citationDetail(1)))
    const onClose = vi.fn<() => void>()
    await renderRoutes(sheetPage(onClose), { initialEntry: "/chat/c1" })
    await expect.element(page.getByText("Toàn bộ mục")).toBeVisible()
    await userEvent.keyboard("{Escape}")
    expect(onClose).toHaveBeenCalledOnce()
  })

  test("uses a drawer on a phone-sized viewport", async () => {
    await page.viewport(375, 800)
    worker.use(
      getGetMessageCitationMockHandler(citationDetail(1, { strategy: "chunk_window" }))
    )
    await renderRoutes(sheetPage(vi.fn<() => void>()), { initialEntry: "/chat/c1" })
    await expect.element(page.getByText("Đoạn và các đoạn lân cận")).toBeVisible()
    expect(document.querySelector('[data-slot="sheet-content"]')).toBeNull()
  })
})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd frontend && npx vitest run --project browser app/features/citations/CitationSheet.browser.test.tsx`
Expected: FAIL with `Failed to resolve import "./CitationSheet"`.

- [ ] **Step 3: Write `CitationSheet.tsx`**

`frontend/app/features/citations/CitationSheet.tsx`:

```tsx
import { useQuery } from "@tanstack/react-query"
import { useEffect, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { Streamdown } from "streamdown"

import { getGetMessageCitationQueryOptions } from "~/api/gen/endpoints"
import type { CitationDetail } from "~/api/gen/schemas"
import { Badge } from "~/components/ui/badge"
import {
  Drawer,
  DrawerContent,
  DrawerHeader,
  DrawerTitle,
} from "~/components/ui/drawer"
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "~/components/ui/sheet"
import { Spinner } from "~/components/ui/spinner"
import { streamdownPlugins } from "~/features/chat/lib/streamdown-config"
import { useIsMobile } from "~/hooks/use-mobile"
import { apiErrorMessage } from "~/i18n/error-message"
import { cn } from "~/lib/utils"

import { usePageText } from "./lib/citation-format"

export type CitationTarget = {
  messageId: string
  index: number
}

export type CitationSheetProps = {
  target: CitationTarget | null
  onClose: () => void
}

const STRATEGY_KEYS = {
  full_section: "strategy.full_section",
  chunk_window: "strategy.chunk_window",
  search_only: "strategy.search_only",
} as const satisfies Record<CitationDetail["strategy"], string>

export function CitationDetailView({ messageId, index }: CitationTarget) {
  const { t } = useTranslation("citations")
  const { t: tErrors } = useTranslation("errors")
  const pageText = usePageText()
  const query = useQuery(getGetMessageCitationQueryOptions(messageId, index))
  const matchedRef = useRef<HTMLElement>(null)

  useEffect(() => {
    matchedRef.current?.scrollIntoView({ block: "center" })
  }, [query.data])

  if (query.isPending) {
    return (
      <div
        role="status"
        className="flex items-center gap-2 text-sm text-muted-foreground"
      >
        <Spinner />
        {t("sheet.loading")}
      </div>
    )
  }

  if (query.isError) {
    return (
      <div role="alert" className="flex flex-col gap-1 text-sm">
        <p className="font-medium text-destructive">{t("sheet.loadFailed")}</p>
        <p className="text-muted-foreground">{apiErrorMessage(query.error, tErrors)}</p>
      </div>
    )
  }

  const detail = query.data
  const pages = pageText(detail.start_page, detail.end_page)
  return (
    <article className="flex flex-col gap-4">
      <header className="flex flex-col gap-1">
        <p className="text-xs text-muted-foreground">{detail.source}</p>
        <p className="text-base font-medium">
          {`${detail.document_title} › ${detail.section}`}
        </p>
        <p className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
          {pages === null ? null : <span>{pages}</span>}
          <Badge variant="outline">{t(STRATEGY_KEYS[detail.strategy])}</Badge>
        </p>
        {detail.is_current ? null : (
          <p className="text-xs text-amber-600 dark:text-amber-400">{t("stale")}</p>
        )}
      </header>
      {detail.chunks.map((chunk) => (
        <section
          key={chunk.id}
          ref={chunk.matched ? matchedRef : undefined}
          data-matched={chunk.matched}
          aria-label={chunk.matched ? t("sheet.matched") : undefined}
          className={cn(
            "rounded-xl px-3 py-2",
            chunk.matched && "bg-amber-500/10 ring-1 ring-amber-500/30"
          )}
        >
          <Streamdown
            mode="static"
            plugins={streamdownPlugins}
            className="text-sm leading-relaxed"
          >
            {chunk.text}
          </Streamdown>
        </section>
      ))}
    </article>
  )
}

export function CitationSheet({ target, onClose }: CitationSheetProps) {
  const { t } = useTranslation("citations")
  const isMobile = useIsMobile()
  const [shown, setShown] = useState(target)
  if (target !== null && target !== shown) {
    setShown(target)
  }

  const open = target !== null
  const title = shown === null ? "" : t("sheet.title", { index: shown.index })
  const body =
    shown === null ? null : (
      <CitationDetailView messageId={shown.messageId} index={shown.index} />
    )
  const handleOpenChange = (next: boolean) => {
    if (!next) {
      onClose()
    }
  }

  if (isMobile) {
    return (
      <Drawer open={open} onOpenChange={handleOpenChange}>
        <DrawerContent className="max-h-[85dvh]">
          <DrawerHeader>
            <DrawerTitle>{title}</DrawerTitle>
          </DrawerHeader>
          <div className="overflow-y-auto px-4 pb-6">{body}</div>
        </DrawerContent>
      </Drawer>
    )
  }

  return (
    <Sheet open={open} onOpenChange={handleOpenChange}>
      <SheetContent side="right" className="w-full data-[side=right]:sm:max-w-xl">
        <SheetHeader>
          <SheetTitle>{title}</SheetTitle>
        </SheetHeader>
        <div className="min-h-0 flex-1 overflow-y-auto px-4 pb-6">{body}</div>
      </SheetContent>
    </Sheet>
  )
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd frontend && npx vitest run --project browser app/features/citations/CitationSheet.browser.test.tsx`
Expected: PASS (4 tests).

- [ ] **Step 5: Full check**

Run: `cd frontend && npm run format:write && npm run lint && npm run format && npm run typecheck && npm test`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add frontend/app/features/citations/CitationSheet.tsx frontend/app/features/citations/CitationSheet.browser.test.tsx
git commit -m "feat(citations): show the full citation block in a sheet or drawer" \
  -m "$COMMIT_TRAILER"
```

---

### Task 8: Conversation cache helpers and feedback actions (spec B §10.7)

**Files:**
- Create: `frontend/app/features/conversations/lib/conversation-cache.ts`, `frontend/app/features/chat/lib/feedback-note.ts`, `frontend/app/features/chat/components/MessageActions.tsx`
- Test: `frontend/app/features/conversations/lib/conversation-cache.test.ts`, `frontend/app/features/chat/lib/feedback-note.test.ts`, `frontend/app/features/chat/components/MessageActions.browser.test.tsx`

**Interfaces:**
- Consumes: `getListConversationsInfiniteQueryKey`, `getListMessagesInfiniteQueryKey`, `useSubmitFeedback` (orval); `submitFeedbackBodyNoteMax` (orval zod); `toast`; shadcn `Dialog*`, `Button`; `FeedbackDialog` (Task 2); fixtures, `renderRoutes` (Task 1); `worker`.
- Produces:
  - `updateConversationTitle(queryClient: QueryClient, conversation: { id: string; title: string }): boolean` (true when a cached item changed)
  - `replaceConversationInCache(queryClient: QueryClient, conversation: ConversationView): void`
  - `removeConversationFromCache(queryClient: QueryClient, conversationId: string): void`
  - `invalidateConversations(queryClient: QueryClient): Promise<void>`
  - `setMessageFeedbackInCache(queryClient: QueryClient, conversationId: string, messageId: string, feedback: MessageFeedback | null): void`
  - `composeFeedbackNote(reasonLabels: readonly string[], note: string): string`
  - `MessageActions(props: { conversationId: string; messageId: string; text: string; feedback: MessageFeedback | null; onFeedbackChange: (messageId: string, feedback: MessageFeedback | null) => void })` with buttons `Sao chép`, `Hữu ích`, `Chưa hữu ích`

The backend stores only `{rating, note}` and overwrites on every call. `Hữu ích` sends `{rating: "up", note: ""}`. `Chưa hữu ích` sends `{rating: "down", note: ""}` immediately and opens `FeedbackDialog`; submitting the dialog sends `{rating: "down", note: "[<reasons>] <note>"}` (cut to `submitFeedbackBodyNoteMax`). Each change is applied optimistically to the `useChat` messages (through `onFeedbackChange`) and to the history query cache, and rolled back with a toast when the request fails.

- [ ] **Step 1: Write the failing unit tests**

`frontend/app/features/conversations/lib/conversation-cache.test.ts`:

```ts
import { type InfiniteData, QueryClient } from "@tanstack/react-query"
import { describe, expect, test } from "vitest"

import {
  getListConversationsInfiniteQueryKey,
  getListMessagesInfiniteQueryKey,
} from "~/api/gen/endpoints"
import type { ConversationPage, MessagePage } from "~/api/gen/schemas"

import {
  apiAssistantMessage,
  conversationPage,
  conversationView,
  messagePage,
} from "../../../../tests/chat/fixtures"

import {
  removeConversationFromCache,
  replaceConversationInCache,
  setMessageFeedbackInCache,
  updateConversationTitle,
} from "./conversation-cache"

type Cursor = string | null | undefined

const LIST_KEY = getListConversationsInfiniteQueryKey({ limit: 20 })

function seedConversations(queryClient: QueryClient) {
  queryClient.setQueryData<InfiniteData<ConversationPage, Cursor>>(LIST_KEY, {
    pages: [
      conversationPage(
        [
          conversationView("c1", "Cuộc trò chuyện mới"),
          conversationView("c2", "Ibuprofen"),
        ],
        "cursor-1"
      ),
      conversationPage([conversationView("c3", "Amoxicillin")]),
    ],
    pageParams: [undefined, "cursor-1"],
  })
}

function titles(queryClient: QueryClient): string[] {
  const data = queryClient.getQueryData<InfiniteData<ConversationPage, Cursor>>(LIST_KEY)
  return data?.pages.flatMap((page) => page.items.map((item) => item.title)) ?? []
}

describe("conversation cache", () => {
  test("updates a title and reports whether it was cached", () => {
    const queryClient = new QueryClient()
    seedConversations(queryClient)
    expect(updateConversationTitle(queryClient, { id: "c1", title: "Liều paracetamol" })).toBe(
      true
    )
    expect(updateConversationTitle(queryClient, { id: "missing", title: "x" })).toBe(false)
    expect(titles(queryClient)).toEqual(["Liều paracetamol", "Ibuprofen", "Amoxicillin"])
  })

  test("replaces and removes conversations across pages", () => {
    const queryClient = new QueryClient()
    seedConversations(queryClient)
    replaceConversationInCache(queryClient, conversationView("c3", "Amoxicillin cho trẻ"))
    removeConversationFromCache(queryClient, "c2")
    expect(titles(queryClient)).toEqual(["Cuộc trò chuyện mới", "Amoxicillin cho trẻ"])
  })

  test("sets feedback on a cached history message", () => {
    const queryClient = new QueryClient()
    const key = getListMessagesInfiniteQueryKey("c1", { limit: 30 })
    queryClient.setQueryData<InfiniteData<MessagePage, Cursor>>(key, {
      pages: [
        messagePage([apiAssistantMessage("a1", "Liều."), apiAssistantMessage("a2", "Khác.")]),
      ],
      pageParams: [undefined],
    })
    setMessageFeedbackInCache(queryClient, "c1", "a1", { rating: "down", note: "Thiếu" })
    const items =
      queryClient.getQueryData<InfiniteData<MessagePage, Cursor>>(key)?.pages[0]?.items ?? []
    expect(items.map((item) => item.metadata.feedback ?? null)).toEqual([
      { rating: "down", note: "Thiếu" },
      null,
    ])
  })
})
```

`frontend/app/features/chat/lib/feedback-note.test.ts`:

```ts
import { describe, expect, test } from "vitest"

import { submitFeedbackBodyNoteMax } from "~/api/gen/zod"

import { composeFeedbackNote } from "./feedback-note"

describe("composeFeedbackNote", () => {
  test("puts selected reasons before the note", () => {
    expect(composeFeedbackNote(["Sai thông tin", "Khó hiểu"], "  Thiếu liều trẻ em ")).toBe(
      "[Sai thông tin, Khó hiểu] Thiếu liều trẻ em"
    )
  })

  test("handles reasons only, note only and nothing", () => {
    expect(composeFeedbackNote(["Sai thông tin"], "")).toBe("[Sai thông tin]")
    expect(composeFeedbackNote([], "Thiếu")).toBe("Thiếu")
    expect(composeFeedbackNote([], "   ")).toBe("")
  })

  test("never exceeds the backend limit", () => {
    expect(composeFeedbackNote(["Sai thông tin"], "x".repeat(5000))).toHaveLength(
      submitFeedbackBodyNoteMax
    )
  })
})
```

- [ ] **Step 2: Write the failing browser test**

`frontend/app/features/chat/components/MessageActions.browser.test.tsx`:

```tsx
import { http, HttpResponse } from "msw"
import { useState } from "react"
import { describe, expect, test, vi } from "vitest"
import { page, userEvent } from "vitest/browser"

import type { MessageFeedback } from "~/api/gen/schemas"

import { renderRoutes } from "../../../../tests/chat/render"
import { worker } from "../../../../tests/msw/browser"

import { MessageActions } from "./MessageActions"

const FEEDBACK_URL = "*/api/v1/messages/:messageId/feedback"

type FeedbackChange = (messageId: string, feedback: MessageFeedback | null) => void

function actionsPage(onChange: FeedbackChange) {
  return function Page() {
    const [feedback, setFeedback] = useState<MessageFeedback | null>(null)
    return (
      <MessageActions
        conversationId="c1"
        messageId="a1"
        text="Người lớn 500 mg [1]."
        feedback={feedback}
        onFeedbackChange={(messageId, next) => {
          onChange(messageId, next)
          setFeedback(next)
        }}
      />
    )
  }
}

function captureFeedback(): unknown[] {
  const bodies: unknown[] = []
  worker.use(
    http.post(FEEDBACK_URL, async ({ request, params }) => {
      bodies.push(await request.json())
      return HttpResponse.json({
        message_id: String(params["messageId"]),
        rating: "up",
        note: "",
      })
    })
  )
  return bodies
}

describe("MessageActions", () => {
  test("sends Hữu ích and marks it pressed", async () => {
    const bodies = captureFeedback()
    const onChange = vi.fn<FeedbackChange>()
    await renderRoutes(actionsPage(onChange), { initialEntry: "/chat/c1" })
    const up = page.getByRole("button", { name: "Hữu ích", exact: true })
    await up.click()
    await expect.element(up).toHaveAttribute("aria-pressed", "true")
    await vi.waitFor(() => expect(bodies).toEqual([{ rating: "up", note: "" }]))
    expect(onChange).toHaveBeenCalledWith("a1", { rating: "up", note: "" })
  })

  test("sends Chưa hữu ích, then the reasons and note from the dialog", async () => {
    const bodies = captureFeedback()
    await renderRoutes(actionsPage(vi.fn<FeedbackChange>()), { initialEntry: "/chat/c1" })
    await page.getByRole("button", { name: "Chưa hữu ích" }).click()
    await page.getByRole("button", { name: "Sai thông tin" }).click()
    await userEvent.fill(page.getByRole("textbox", { name: "Ghi chú" }), "Thiếu liều trẻ em")
    await page.getByRole("button", { name: "Gửi phản hồi" }).click()
    await expect.element(page.getByText("Cảm ơn bạn đã góp ý.")).toBeVisible()
    await vi.waitFor(() =>
      expect(bodies).toEqual([
        { rating: "down", note: "" },
        { rating: "down", note: "[Sai thông tin] Thiếu liều trẻ em" },
      ])
    )
  })

  test("rolls back and shows a toast when the request fails", async () => {
    worker.use(
      http.post(FEEDBACK_URL, () =>
        HttpResponse.json(
          {
            type: "urn:pharma-agent:problem:message-not-found",
            title: "Message not found",
            status: 404,
            code: "MESSAGE_NOT_FOUND",
          },
          { status: 404, headers: { "Content-Type": "application/problem+json" } }
        )
      )
    )
    const onChange = vi.fn<FeedbackChange>()
    await renderRoutes(actionsPage(onChange), { initialEntry: "/chat/c1" })
    const up = page.getByRole("button", { name: "Hữu ích", exact: true })
    await up.click()
    await expect.element(page.getByText("Không gửi được đánh giá")).toBeVisible()
    await expect.element(up).toHaveAttribute("aria-pressed", "false")
    expect(onChange).toHaveBeenLastCalledWith("a1", null)
  })

  test("copies the answer text", async () => {
    const writeText = vi
      .spyOn(navigator.clipboard, "writeText")
      .mockResolvedValue(undefined)
    await renderRoutes(actionsPage(vi.fn<FeedbackChange>()), { initialEntry: "/chat/c1" })
    await page.getByRole("button", { name: "Sao chép" }).click()
    expect(writeText).toHaveBeenCalledWith("Người lớn 500 mg [1].")
    await expect.element(page.getByRole("button", { name: "Đã sao chép" })).toBeVisible()
  })
})
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cd frontend && npx vitest run app/features/conversations/lib/conversation-cache.test.ts app/features/chat/lib/feedback-note.test.ts app/features/chat/components/MessageActions.browser.test.tsx`
Expected: FAIL with `Failed to resolve import "./conversation-cache"`, `"./feedback-note"` and `"./MessageActions"`.

- [ ] **Step 4: Write the cache helpers and the note composer**

`frontend/app/features/conversations/lib/conversation-cache.ts`:

```ts
import type { InfiniteData, QueryClient } from "@tanstack/react-query"

import {
  getListConversationsInfiniteQueryKey,
  getListMessagesInfiniteQueryKey,
} from "~/api/gen/endpoints"
import type {
  ConversationPage,
  ConversationView,
  MessageFeedback,
  MessagePage,
} from "~/api/gen/schemas"

type Cursor = string | null | undefined

function mapConversationItems(
  queryClient: QueryClient,
  update: (items: ConversationView[]) => ConversationView[]
): void {
  queryClient.setQueriesData<InfiniteData<ConversationPage, Cursor>>(
    { queryKey: getListConversationsInfiniteQueryKey() },
    (data) =>
      data === undefined
        ? data
        : {
            ...data,
            pages: data.pages.map((page) => ({ ...page, items: update(page.items) })),
          }
  )
}

export function updateConversationTitle(
  queryClient: QueryClient,
  conversation: { id: string; title: string }
): boolean {
  let found = false
  mapConversationItems(queryClient, (items) =>
    items.map((item) => {
      if (item.id !== conversation.id) {
        return item
      }
      found = true
      return { ...item, title: conversation.title }
    })
  )
  return found
}

export function replaceConversationInCache(
  queryClient: QueryClient,
  conversation: ConversationView
): void {
  mapConversationItems(queryClient, (items) =>
    items.map((item) => (item.id === conversation.id ? conversation : item))
  )
}

export function removeConversationFromCache(
  queryClient: QueryClient,
  conversationId: string
): void {
  mapConversationItems(queryClient, (items) =>
    items.filter((item) => item.id !== conversationId)
  )
}

export function invalidateConversations(queryClient: QueryClient): Promise<void> {
  return queryClient.invalidateQueries({
    queryKey: getListConversationsInfiniteQueryKey(),
  })
}

export function setMessageFeedbackInCache(
  queryClient: QueryClient,
  conversationId: string,
  messageId: string,
  feedback: MessageFeedback | null
): void {
  queryClient.setQueriesData<InfiniteData<MessagePage, Cursor>>(
    { queryKey: getListMessagesInfiniteQueryKey(conversationId) },
    (data) =>
      data === undefined
        ? data
        : {
            ...data,
            pages: data.pages.map((page) => ({
              ...page,
              items: page.items.map((item) =>
                item.id === messageId
                  ? { ...item, metadata: { ...item.metadata, feedback } }
                  : item
              ),
            })),
          }
  )
}
```

`frontend/app/features/chat/lib/feedback-note.ts`:

```ts
import { submitFeedbackBodyNoteMax } from "~/api/gen/zod"

export function composeFeedbackNote(
  reasonLabels: readonly string[],
  note: string
): string {
  const trimmed = note.trim()
  const prefix = reasonLabels.length === 0 ? "" : `[${reasonLabels.join(", ")}]`
  let combined = trimmed
  if (prefix !== "") {
    combined = trimmed === "" ? prefix : `${prefix} ${trimmed}`
  }
  return combined.slice(0, submitFeedbackBodyNoteMax)
}
```

- [ ] **Step 5: Write `MessageActions.tsx`**

`frontend/app/features/chat/components/MessageActions.tsx`:

```tsx
import { useQueryClient } from "@tanstack/react-query"
import { CheckIcon, CopyIcon, ThumbsDownIcon, ThumbsUpIcon } from "lucide-react"
import { useEffect, useMemo, useState } from "react"
import { useTranslation } from "react-i18next"

import { useSubmitFeedback } from "~/api/gen/endpoints"
import type { MessageFeedback } from "~/api/gen/schemas"
import {
  FeedbackDialog,
  type FeedbackReason,
} from "~/components/elements/feedback-dialog"
import { Button } from "~/components/ui/button"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "~/components/ui/dialog"
import { toast } from "~/components/ui/toast"
import { setMessageFeedbackInCache } from "~/features/conversations/lib/conversation-cache"
import { cn } from "~/lib/utils"

import { composeFeedbackNote } from "../lib/feedback-note"

export type MessageActionsProps = {
  conversationId: string
  messageId: string
  text: string
  feedback: MessageFeedback | null
  onFeedbackChange: (messageId: string, feedback: MessageFeedback | null) => void
}

const COPIED_RESET_MS = 2000

export function MessageActions({
  conversationId,
  messageId,
  text,
  feedback,
  onFeedbackChange,
}: MessageActionsProps) {
  const { t } = useTranslation("chat")
  const queryClient = useQueryClient()
  const submitFeedback = useSubmitFeedback()
  const [copied, setCopied] = useState(false)
  const [dialogOpen, setDialogOpen] = useState(false)
  const [selected, setSelected] = useState<string[]>([])
  const [note, setNote] = useState("")
  const [sent, setSent] = useState(false)

  const reasons = useMemo<FeedbackReason[]>(
    () => [
      { id: "incorrect", label: t("feedback.reasons.incorrect") },
      { id: "incomplete", label: t("feedback.reasons.incomplete") },
      { id: "wrongSource", label: t("feedback.reasons.wrongSource") },
      { id: "unclear", label: t("feedback.reasons.unclear") },
    ],
    [t]
  )

  useEffect(() => {
    if (!copied) {
      return undefined
    }
    const timer = window.setTimeout(() => setCopied(false), COPIED_RESET_MS)
    return () => window.clearTimeout(timer)
  }, [copied])

  function apply(next: MessageFeedback) {
    const previous = feedback
    onFeedbackChange(messageId, next)
    setMessageFeedbackInCache(queryClient, conversationId, messageId, next)
    submitFeedback.mutate(
      { messageId, data: next },
      {
        onError: () => {
          onFeedbackChange(messageId, previous)
          setMessageFeedbackInCache(queryClient, conversationId, messageId, previous)
          toast.add({ title: t("actions.feedbackFailed"), type: "error" })
        },
      }
    )
  }

  async function copy() {
    await navigator.clipboard.writeText(text)
    setCopied(true)
  }

  function openNotHelpful() {
    setSelected([])
    setNote("")
    setSent(false)
    setDialogOpen(true)
    apply({ rating: "down", note: "" })
  }

  function toggleReason(id: string) {
    setSelected((current) =>
      current.includes(id) ? current.filter((value) => value !== id) : [...current, id]
    )
  }

  function sendDetails() {
    const labels = reasons
      .filter((reason) => selected.includes(reason.id))
      .map((reason) => reason.label)
    apply({ rating: "down", note: composeFeedbackNote(labels, note) })
    setSent(true)
  }

  return (
    <div className="flex items-center gap-0.5">
      <Button
        variant="ghost"
        size="icon-sm"
        aria-label={copied ? t("actions.copied") : t("actions.copy")}
        onClick={() => void copy()}
      >
        {copied ? <CheckIcon aria-hidden /> : <CopyIcon aria-hidden />}
      </Button>
      <Button
        variant="ghost"
        size="icon-sm"
        aria-label={t("actions.up")}
        aria-pressed={feedback?.rating === "up"}
        onClick={() => apply({ rating: "up", note: "" })}
      >
        <ThumbsUpIcon
          aria-hidden
          className={cn(feedback?.rating === "up" && "fill-current")}
        />
      </Button>
      <Button
        variant="ghost"
        size="icon-sm"
        aria-label={t("actions.down")}
        aria-pressed={feedback?.rating === "down"}
        onClick={openNotHelpful}
      >
        <ThumbsDownIcon
          aria-hidden
          className={cn(feedback?.rating === "down" && "fill-current")}
        />
      </Button>
      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader className="sr-only">
            <DialogTitle>{t("feedback.title")}</DialogTitle>
          </DialogHeader>
          <FeedbackDialog
            reasons={reasons}
            selected={selected}
            note={note}
            sent={sent}
            onToggleReason={toggleReason}
            onNoteChange={setNote}
            onSubmit={sendDetails}
          />
        </DialogContent>
      </Dialog>
    </div>
  )
}
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cd frontend && npx vitest run app/features/conversations/lib/conversation-cache.test.ts app/features/chat/lib/feedback-note.test.ts app/features/chat/components/MessageActions.browser.test.tsx`
Expected: PASS (3 + 3 unit tests, 4 browser tests).

- [ ] **Step 7: Full check**

Run: `cd frontend && npm run format:write && npm run lint && npm run format && npm run typecheck && npm test`
Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add frontend/app/features/conversations/lib/conversation-cache.ts frontend/app/features/conversations/lib/conversation-cache.test.ts \
  frontend/app/features/chat/lib/feedback-note.ts frontend/app/features/chat/lib/feedback-note.test.ts \
  frontend/app/features/chat/components/MessageActions.tsx frontend/app/features/chat/components/MessageActions.browser.test.tsx
git commit -m "feat(chat): add feedback actions and conversation cache helpers" \
  -m "$COMMIT_TRAILER"
```

---

### Task 9: `ChatThread`, `ChatSession`, history paging, error table and the chat route (spec B §10.1, §10.3, §10.4, §10.8)

**Files:**
- Create: `frontend/app/features/chat/ChatThread.tsx`, `frontend/app/features/chat/components/ChatSession.tsx`, `frontend/app/features/chat/components/Composer.tsx`, `frontend/app/features/chat/components/OlderMessagesLoader.tsx`, `frontend/app/features/chat/components/AgentUnavailableBanner.tsx`, `frontend/app/features/chat/lib/location-state.ts`, `frontend/tests/chat/stream-handler.ts`
- Modify (replace P8's placeholder): `frontend/app/routes/app/chat.tsx`
- Test: `frontend/app/features/chat/ChatThread.browser.test.tsx`, `frontend/app/features/chat/history-prepend.browser.test.tsx`

**Interfaces:**
- Consumes: `useListMessagesInfinite`, `getListConversationsInfiniteQueryKey` (orval); `getListMessagesMockHandler`, `getGetMessageCitationMockHandler` (`~/api/gen/endpoints.msw`); `chatStreamBodyMessageMax` (orval zod); `isApiError`; `apiErrorMessage`; `toast`; shadcn `MessageScroller*`, `InputGroup*`, `Alert*`, `Empty*`, `Spinner`, `Button`; `@shadcn/helpers/ai-sdk` `createChat`; everything from Tasks 1–8.
- Produces:
  - `HISTORY_PAGE_SIZE = 30`; `ChatThread({ conversationId }: { conversationId: string | undefined })` (pinned entry); `ChatThreadFallback()`
  - `ChatSession(props: ChatSessionProps)` with `type ChatSessionProps = { conversationId: string; initialMessages: PharmaUIMessage[]; pages: readonly MessagePage[]; hasOlder: boolean; loadingOlder: boolean; onLoadOlder: () => void; initialQuestion: string | undefined; onInitialQuestionSent: () => void }`
  - `Composer(props: { value: string; onValueChange: (value: string) => void; busy: boolean; error: string | null; onSubmit: (text: string) => void; onStop: () => void; disabled?: boolean })` (textbox `Tin nhắn`, buttons `Gửi` / `Dừng`)
  - `OlderMessagesLoader(props: { hasOlder: boolean; loadingOlder: boolean; userScrolled: boolean; onLoadOlder: () => void })`
  - `AgentUnavailableBanner()`
  - `type ChatLocationState = { initialQuestion: string }`, `readInitialQuestion(state: unknown): string | undefined`
  - `tests/chat/stream-handler.ts`: `STREAM_URL`, `type StreamRequest`, `type ScriptedChat`, `finishMetadata(overrides?)`, `answerChat(question, options?)`, `chatStreamHandler(chats, requests?, delayMs?)`, `problemHandler(status, code, detail?)`, `networkErrorHandler()`

Behaviour:

- `ChatThread` keys `ConversationChat` by `conversationId`, loads the first history page (`limit=30`, `staleTime: Infinity`, `gcTime: 0` so a revisit fetches fresh history) and mounts `ChatSession` only after that page exists, because `useChat` reads `messages` once when it creates its `Chat` and recreates (and stops) it when `id` changes.
- Each rendered message is a `MessageScrollerItem` with `messageId` and `data-message-id`; user messages are scroll anchors (`defaultScrollPosition="last-anchor"`).
- Older pages: `OlderMessagesLoader` calls `onLoadOlder` when `useMessageScrollerScrollable().start === false` (top reached) and either the user scrolled (wheel, touch or key on the viewport) or the content does not scroll (`end === false`). `ChatSession` prepends each new page once (`olderMessagesFrom` + `prependOlder`); `MessageScrollerViewport` keeps `preserveScrollOnPrepend` on.
- Error table (spec B §10.8):

| Case | Detection | Handling |
| --- | --- | --- |
| 401 | stream problem `status === 401` | drop the unsent turn, `navigate("/login?next=<encoded path+search>")` (history 401 goes through P8's QueryClient redirect) |
| 404 `CONVERSATION_NOT_FOUND` | stream problem code, or `ApiError` from the history query | toast with the translated code, `navigate("/chat", { replace: true })` |
| 422 | Composer blocks more than `chatStreamBodyMessageMax` characters; stream problem `status === 422` | drop the unsent turn, restore the draft, show `detail` (or the translated code) under the composer |
| 503 (`AGENT_UNAVAILABLE`, `SERVICE_STARTING`, …) | stream problem `status === 503` | drop the unsent turn, restore the draft, show `AgentUnavailableBanner` |
| Stop | `stop()`; `onFinish` `isAbort` | message shows `Marker` "Đã dừng" (the backend cancels the graph and does not persist) |
| Network loss | `onError` without a problem body; `onFinish` `isError`/`isDisconnect` | `ErrorState` on the failed message, or after the last question when no assistant message started; retry drops the failed turn and resends the question |
| `status ∈ {error, timeout}` | finish metadata | `ErrorState` from `MessageParts`; retry resends the question before that message |
| `persisted === false` | finish metadata | toast "Không lưu được lượt này" |

- `data-conversation` (transient) updates the title in the conversation list cache; when the conversation is not cached yet (a new conversation has `turn_count = 0` and is not listed) the list is invalidated. Every finished turn invalidates the conversation list.

- [ ] **Step 1: Write the stream test support**

`frontend/tests/chat/stream-handler.ts`:

```ts
import { createChat } from "@shadcn/helpers/ai-sdk"
import { createUIMessageStreamResponse } from "ai"
import { type HttpHandler, http, HttpResponse } from "msw"
import { z } from "zod"

import type { MessageMetadata } from "~/api/gen/schemas"
import { CHAT_STREAM_API } from "~/features/chat/lib/chat-transport"
import type { PharmaUIMessage } from "~/features/chat/lib/message-schema"

import { CREATED_AT, pharmaSource } from "./fixtures"

const streamRequestSchema = z.object({
  conversation_id: z.string(),
  message: z.string(),
})

export type StreamRequest = z.infer<typeof streamRequestSchema>
export type ScriptedChat = ReturnType<typeof createChat<PharmaUIMessage>>

export const STREAM_URL = `*${CHAT_STREAM_API}`

export function finishMetadata(overrides: Partial<MessageMetadata> = {}): MessageMetadata {
  return {
    status: "completed",
    createdAt: CREATED_AT,
    errorCode: null,
    usage: { llmCalls: 4, promptTokens: 900, completionTokens: 120, searchRounds: 1 },
    runId: "00000000000000000000000000000003",
    persisted: true,
    ...overrides,
  }
}

export type AnswerOptions = {
  id?: string
  conversationId?: string
  title?: string
  text?: string
  metadata?: Partial<MessageMetadata>
  slow?: boolean
  withSource?: boolean
}

export function answerChat(question: string, options: AnswerOptions = {}): ScriptedChat {
  const text =
    options.text ?? "Người lớn uống 0,5–1 g mỗi 4–6 giờ, tối đa 4 g mỗi ngày [1]."
  return createChat<PharmaUIMessage>()
    .user(question)
    .assistant(
      ({ writer }) => {
        writer.data({
          type: "data-conversation",
          data: {
            id: options.conversationId ?? "c1",
            title: options.title ?? "Cuộc trò chuyện mới",
          },
          transient: true,
        })
        writer.data({
          type: "data-phase",
          id: "phase",
          data: { phase: "searching", round: 1 },
        })
        if (options.slow === true) {
          writer.sleep(600)
        }
        writer.data({
          type: "data-skills",
          data: { skills: [{ name: "drug-monograph", title: "Chuyên luận thuốc" }] },
        })
        writer.data({ type: "data-phase", id: "phase", data: { phase: "answering" } })
        writer.data({
          type: "data-evidence",
          id: "evidence",
          data: {
            items: [
              {
                index: 1,
                source: "Dược thư Quốc gia Việt Nam (Xuất bản lần thứ 2)",
                title: "Paracetamol",
                section: "Liều lượng và cách dùng",
                startPage: 812,
                endPage: 813,
                snippet: "Người lớn và trẻ em trên 12 tuổi…",
              },
            ],
          },
        })
        if (text !== "") {
          if (options.slow === true) {
            writer.text(text, { delayMs: 150 })
          } else {
            writer.text(text)
          }
        }
        if (options.withSource !== false) {
          writer.sourceDocument({
            sourceId: "chunk-1",
            mediaType: "text/markdown",
            title: "Paracetamol › Liều lượng và cách dùng",
            providerMetadata: { pharma: { ...pharmaSource(1) } },
          })
        }
      },
      { id: options.id ?? "a-new", metadata: finishMetadata(options.metadata) }
    )
}

export function chatStreamHandler(
  chats: readonly ScriptedChat[],
  requests: StreamRequest[] = [],
  delayMs = 0
): HttpHandler {
  let served = 0
  return http.post(STREAM_URL, async ({ request }) => {
    const body = streamRequestSchema.parse(await request.json())
    requests.push(body)
    const chat = chats[Math.min(served, chats.length - 1)]
    served += 1
    if (chat === undefined) {
      return HttpResponse.error()
    }
    const stream = await chat.transport({ delayMs }).sendMessages({
      trigger: "submit-message",
      chatId: body.conversation_id,
      messageId: undefined,
      messages: chat.get(1),
      abortSignal: request.signal,
    })
    return createUIMessageStreamResponse({ stream })
  })
}

export function problemHandler(
  status: number,
  code: string,
  detail?: string
): HttpHandler {
  return http.post(STREAM_URL, () =>
    HttpResponse.json(
      {
        type: `urn:pharma-agent:problem:${code.toLowerCase().replaceAll("_", "-")}`,
        title: code,
        status,
        code,
        ...(detail === undefined ? {} : { detail }),
      },
      { status, headers: { "Content-Type": "application/problem+json" } }
    )
  )
}

export function networkErrorHandler(): HttpHandler {
  return http.post(STREAM_URL, () => HttpResponse.error())
}
```

- [ ] **Step 2: Write the failing browser tests**

`frontend/app/features/chat/ChatThread.browser.test.tsx`:

```tsx
import type { InfiniteData } from "@tanstack/react-query"
import { http, HttpResponse } from "msw"
import { useParams } from "react-router"
import { describe, expect, test, vi } from "vitest"
import { page, userEvent } from "vitest/browser"

import { getListConversationsInfiniteQueryKey } from "~/api/gen/endpoints"
import {
  getGetMessageCitationMockHandler,
  getListMessagesMockHandler,
} from "~/api/gen/endpoints.msw"
import type { ConversationPage } from "~/api/gen/schemas"

import {
  apiAssistantMessage,
  apiUserMessage,
  citationDetail,
  conversationPage,
  conversationView,
  messagePage,
} from "../../../tests/chat/fixtures"
import { renderRoutes } from "../../../tests/chat/render"
import {
  answerChat,
  chatStreamHandler,
  networkErrorHandler,
  problemHandler,
  type StreamRequest,
} from "../../../tests/chat/stream-handler"
import { worker } from "../../../tests/msw/browser"

import { ChatThread } from "./ChatThread"

const QUESTION = "Liều paracetamol cho người lớn?"

function ChatPage() {
  const { conversationId } = useParams()
  return (
    <div className="h-[640px]">
      <ChatThread conversationId={conversationId} />
    </div>
  )
}

function withHistory() {
  worker.use(
    getListMessagesMockHandler(
      messagePage([apiUserMessage("u0", "Xin chào"), apiAssistantMessage("a0", "Chào bạn.")])
    )
  )
}

async function openThread() {
  const result = await renderRoutes(ChatPage, { initialEntry: "/chat/c1" })
  await expect.element(page.getByText("Chào bạn.")).toBeVisible()
  return result
}

async function ask(question = QUESTION) {
  await userEvent.fill(page.getByRole("textbox", { name: "Tin nhắn" }), question)
  await page.getByRole("button", { name: "Gửi" }).click()
}

const marker = () => page.getByRole("button", { name: "1", exact: true })

describe("ChatThread", () => {
  test("streams phase, answer and sources after the history, then offers feedback", async () => {
    withHistory()
    const requests: StreamRequest[] = []
    worker.use(chatStreamHandler([answerChat(QUESTION, { slow: true })], requests))
    await openThread()
    await ask()
    await expect.element(page.getByText("Đang tìm kiếm · vòng 1")).toBeVisible()
    await expect.element(marker()).toBeVisible()
    await expect.element(page.getByRole("button", { name: "1 nguồn" })).toBeVisible()
    await expect
      .element(page.getByRole("button", { name: "Hữu ích", exact: true }).nth(1))
      .toBeVisible()
    await expect.element(page.getByText("Đang tìm kiếm · vòng 1")).not.toBeInTheDocument()
    expect(requests).toEqual([{ conversation_id: "c1", message: QUESTION }])
    expect(document.querySelector('[data-message-id="a-new"]')).not.toBeNull()
  })

  test("opens the citation sheet from a marker", async () => {
    withHistory()
    worker.use(
      chatStreamHandler([answerChat(QUESTION)]),
      getGetMessageCitationMockHandler(citationDetail(1))
    )
    await openThread()
    await ask()
    await marker().click()
    await expect.element(page.getByRole("heading", { name: "Nguồn [1]" })).toBeVisible()
    await expect
      .element(page.getByRole("region", { name: "Đoạn được trích dẫn" }))
      .toBeVisible()
  })

  test("shows a guardrail notice for a blocked turn", async () => {
    withHistory()
    worker.use(
      chatStreamHandler([
        answerChat(QUESTION, {
          text: "Mình không thể hỗ trợ yêu cầu này.",
          metadata: { status: "blocked" },
          withSource: false,
        }),
      ])
    )
    await openThread()
    await ask()
    await expect.element(page.getByText("Không thể trả lời câu hỏi này")).toBeVisible()
    await expect.element(page.getByText("Mình không thể hỗ trợ yêu cầu này.")).toBeVisible()
  })

  test("retries a timed out turn with the same question", async () => {
    withHistory()
    const requests: StreamRequest[] = []
    worker.use(
      chatStreamHandler(
        [
          answerChat(QUESTION, {
            id: "a-timeout",
            text: "",
            metadata: { status: "timeout" },
            withSource: false,
          }),
          answerChat(QUESTION, { id: "a-retry" }),
        ],
        requests
      )
    )
    await openThread()
    await ask()
    await expect.element(page.getByText("Agent phản hồi quá lâu.")).toBeVisible()
    await page.getByRole("button", { name: "Thử lại" }).click()
    await expect.element(marker()).toBeVisible()
    expect(requests.map((request) => request.message)).toEqual([QUESTION, QUESTION])
  })

  test("warns when the turn was not persisted and hides feedback for it", async () => {
    withHistory()
    worker.use(chatStreamHandler([answerChat(QUESTION, { metadata: { persisted: false } })]))
    await openThread()
    await ask()
    await expect.element(page.getByText("Không lưu được lượt này")).toBeVisible()
    await expect.element(marker()).toBeVisible()
    expect(
      page.getByRole("button", { name: "Hữu ích", exact: true }).elements()
    ).toHaveLength(1)
  })

  test("stops a streaming turn and marks it", async () => {
    withHistory()
    worker.use(chatStreamHandler([answerChat(QUESTION, { slow: true })]))
    await openThread()
    await ask()
    await expect.element(page.getByText("Đang tìm kiếm · vòng 1")).toBeVisible()
    await page.getByRole("button", { name: "Dừng" }).click()
    await expect.element(page.getByText("Đã dừng")).toBeVisible()
    await expect.element(page.getByRole("button", { name: "Gửi" })).toBeVisible()
  })

  test("sends the user to login on 401", async () => {
    withHistory()
    worker.use(problemHandler(401, "UNAUTHORIZED"))
    await openThread()
    await ask()
    await expect
      .element(page.getByTestId("location"))
      .toHaveTextContent("/login?next=%2Fchat%2Fc1")
  })

  test("returns to /chat with a toast when the conversation is gone", async () => {
    withHistory()
    worker.use(problemHandler(404, "CONVERSATION_NOT_FOUND"))
    await openThread()
    await ask()
    await expect.element(page.getByText("Không tìm thấy cuộc trò chuyện")).toBeVisible()
    await expect.element(page.getByTestId("location")).toHaveTextContent(/^\/chat$/)
  })

  test("restores the draft and shows the validation detail on 422", async () => {
    withHistory()
    worker.use(problemHandler(422, "VALIDATION_ERROR", "Câu hỏi quá dài"))
    await openThread()
    await ask()
    await expect.element(page.getByText("Câu hỏi quá dài")).toBeVisible()
    await expect
      .element(page.getByRole("textbox", { name: "Tin nhắn" }))
      .toHaveValue(QUESTION)
    expect(page.getByText(QUESTION, { exact: true }).elements()).toHaveLength(0)
  })

  test("blocks a question over the length limit before sending", async () => {
    withHistory()
    await openThread()
    await userEvent.fill(page.getByRole("textbox", { name: "Tin nhắn" }), "x".repeat(4001))
    await expect.element(page.getByText("Câu hỏi dài 4001/4000 ký tự")).toBeVisible()
    await expect.element(page.getByRole("button", { name: "Gửi" })).toBeDisabled()
  })

  test("shows the agent unavailable banner on 503", async () => {
    withHistory()
    worker.use(problemHandler(503, "AGENT_UNAVAILABLE"))
    await openThread()
    await ask()
    await expect.element(page.getByText("Agent tạm thời không sẵn sàng")).toBeVisible()
    await expect
      .element(page.getByRole("textbox", { name: "Tin nhắn" }))
      .toHaveValue(QUESTION)
  })

  test("offers a retry after a network failure", async () => {
    withHistory()
    worker.use(networkErrorHandler())
    await openThread()
    await ask()
    await expect.element(page.getByText("Mất kết nối trong lúc trả lời.")).toBeVisible()
    worker.use(chatStreamHandler([answerChat(QUESTION)]))
    await page.getByRole("button", { name: "Thử lại" }).click()
    await expect.element(marker()).toBeVisible()
    expect(page.getByText(QUESTION, { exact: true }).elements()).toHaveLength(1)
  })

  test("sends the question passed in navigation state exactly once", async () => {
    const requests: StreamRequest[] = []
    worker.use(
      getListMessagesMockHandler(messagePage([])),
      chatStreamHandler([answerChat(QUESTION)], requests)
    )
    await renderRoutes(ChatPage, {
      initialEntry: { pathname: "/chat/c1", state: { initialQuestion: QUESTION } },
    })
    await expect.element(marker()).toBeVisible()
    expect(requests).toEqual([{ conversation_id: "c1", message: QUESTION }])
  })

  test("leaves a conversation whose history is not found", async () => {
    worker.use(
      http.get("*/api/v1/conversations/:conversationId/messages", () =>
        HttpResponse.json(
          {
            type: "urn:pharma-agent:problem:conversation-not-found",
            title: "Conversation not found",
            status: 404,
            code: "CONVERSATION_NOT_FOUND",
          },
          { status: 404, headers: { "Content-Type": "application/problem+json" } }
        )
      )
    )
    await renderRoutes(ChatPage, { initialEntry: "/chat/missing" })
    await expect.element(page.getByText("Không tìm thấy cuộc trò chuyện")).toBeVisible()
    await expect.element(page.getByTestId("location")).toHaveTextContent(/^\/chat$/)
  })

  test("updates the list title from the transient data-conversation part", async () => {
    withHistory()
    worker.use(
      chatStreamHandler([
        answerChat(QUESTION, { conversationId: "c1", title: "Liều paracetamol" }),
      ])
    )
    const { queryClient } = await openThread()
    const key = getListConversationsInfiniteQueryKey({ limit: 20 })
    queryClient.setQueryData<InfiniteData<ConversationPage, string | null | undefined>>(
      key,
      {
        pages: [conversationPage([conversationView("c1", "Cuộc trò chuyện mới")])],
        pageParams: [undefined],
      }
    )
    await ask()
    await vi.waitFor(() =>
      expect(
        queryClient.getQueryData<InfiniteData<ConversationPage, string | null | undefined>>(
          key
        )?.pages[0]?.items[0]?.title
      ).toBe("Liều paracetamol")
    )
  })
})
```

`frontend/app/features/chat/history-prepend.browser.test.tsx`:

```tsx
import { http, HttpResponse } from "msw"
import { useParams } from "react-router"
import { describe, expect, test, vi } from "vitest"
import { page, userEvent } from "vitest/browser"

import { conversationTurns, messagePage } from "../../../tests/chat/fixtures"
import { renderRoutes } from "../../../tests/chat/render"
import { worker } from "../../../tests/msw/browser"

import { ChatThread } from "./ChatThread"

function ChatPage() {
  const { conversationId } = useParams()
  return (
    <div className="h-[640px]">
      <ChatThread conversationId={conversationId} />
    </div>
  )
}

function messageElement(id: string): HTMLElement | null {
  return document.querySelector<HTMLElement>(`[data-message-id="${id}"]`)
}

describe("loading older messages", () => {
  test("prepends the older page while the reading position stays", async () => {
    let releaseOlder: () => void = () => undefined
    const olderGate = new Promise<void>((resolve) => {
      releaseOlder = resolve
    })
    const cursors: (string | null)[] = []
    worker.use(
      http.get("*/api/v1/conversations/:conversationId/messages", async ({ request }) => {
        const cursor = new URL(request.url).searchParams.get("cursor")
        cursors.push(cursor)
        if (cursor === null) {
          return HttpResponse.json(
            messagePage(conversationTurns(15, "new"), "cursor-older")
          )
        }
        await olderGate
        return HttpResponse.json(messagePage(conversationTurns(5, "old")))
      })
    )

    await renderRoutes(ChatPage, { initialEntry: "/chat/c1" })
    await expect
      .element(page.getByText("Câu hỏi new 15", { exact: true }))
      .toBeVisible()

    const viewport = document.querySelector<HTMLElement>(
      '[data-slot="message-scroller-viewport"]'
    )
    expect(viewport).not.toBeNull()
    if (viewport === null) {
      return
    }
    await userEvent.wheel(viewport, { delta: { y: -200_000 } })
    await vi.waitFor(() => {
      expect(viewport.scrollTop).toBe(0)
      expect(cursors).toContain("cursor-older")
    })

    const anchor = messageElement("new-u1")
    expect(anchor).not.toBeNull()
    if (anchor === null) {
      return
    }
    const topBefore = anchor.getBoundingClientRect().top
    releaseOlder()

    await vi.waitFor(() => expect(messageElement("old-a5")).not.toBeNull())
    await vi.waitFor(() => {
      expect(Math.abs(anchor.getBoundingClientRect().top - topBefore)).toBeLessThanOrEqual(2)
      expect(viewport.scrollTop).toBeGreaterThan(0)
    })
    expect(messageElement("new-u1")).toBe(anchor)
    expect(cursors.filter((cursor) => cursor === "cursor-older")).toHaveLength(1)
  })
})
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cd frontend && npx vitest run --project browser app/features/chat/ChatThread.browser.test.tsx app/features/chat/history-prepend.browser.test.tsx`
Expected: FAIL with `Failed to resolve import "./ChatThread"`.

- [ ] **Step 4: Write the small pieces**

`frontend/app/features/chat/lib/location-state.ts`:

```ts
import { z } from "zod"

const chatLocationStateSchema = z.object({ initialQuestion: z.string().min(1) })

export type ChatLocationState = z.infer<typeof chatLocationStateSchema>

export function readInitialQuestion(state: unknown): string | undefined {
  const parsed = chatLocationStateSchema.safeParse(state)
  return parsed.success ? parsed.data.initialQuestion : undefined
}
```

`frontend/app/features/chat/components/Composer.tsx`:

```tsx
import { ArrowUpIcon, SquareIcon } from "lucide-react"
import { type FormEvent, type KeyboardEvent, useId } from "react"
import { useTranslation } from "react-i18next"

import { chatStreamBodyMessageMax } from "~/api/gen/zod"
import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupTextarea,
} from "~/components/ui/input-group"
import { cn } from "~/lib/utils"

export type ComposerProps = {
  value: string
  onValueChange: (value: string) => void
  busy: boolean
  error: string | null
  onSubmit: (text: string) => void
  onStop: () => void
  disabled?: boolean
}

export function Composer({
  value,
  onValueChange,
  busy,
  error,
  onSubmit,
  onStop,
  disabled = false,
}: ComposerProps) {
  const { t } = useTranslation("chat")
  const feedbackId = useId()
  const question = value.trim()
  const tooLong = value.length > chatStreamBodyMessageMax
  const canSend = !busy && !disabled && question.length > 0 && !tooLong
  const message = tooLong
    ? t("composer.tooLong", { count: value.length, max: chatStreamBodyMessageMax })
    : error

  function submit() {
    if (canSend) {
      onSubmit(question)
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    submit()
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault()
      submit()
    }
  }

  return (
    <form onSubmit={handleSubmit} className="mx-auto w-full max-w-3xl px-4 pb-4">
      <InputGroup>
        <InputGroupTextarea
          aria-label={t("composer.label")}
          aria-invalid={message !== null}
          aria-describedby={feedbackId}
          placeholder={t("composer.placeholder")}
          rows={2}
          value={value}
          onChange={(event) => onValueChange(event.target.value)}
          onKeyDown={handleKeyDown}
        />
        <InputGroupAddon align="block-end">
          <span
            id={feedbackId}
            className={cn("text-xs", message !== null && "text-destructive")}
          >
            {message}
          </span>
          {busy ? (
            <InputGroupButton
              type="button"
              size="icon-sm"
              variant="default"
              className="ms-auto"
              aria-label={t("composer.stop")}
              onClick={onStop}
            >
              <SquareIcon aria-hidden />
            </InputGroupButton>
          ) : (
            <InputGroupButton
              type="submit"
              size="icon-sm"
              variant="default"
              className="ms-auto"
              aria-label={t("composer.send")}
              disabled={!canSend}
            >
              <ArrowUpIcon aria-hidden />
            </InputGroupButton>
          )}
        </InputGroupAddon>
      </InputGroup>
    </form>
  )
}
```

`frontend/app/features/chat/components/OlderMessagesLoader.tsx`:

```tsx
import { useEffect } from "react"
import { useTranslation } from "react-i18next"

import { useMessageScrollerScrollable } from "~/components/ui/message-scroller"
import { Spinner } from "~/components/ui/spinner"

export type OlderMessagesLoaderProps = {
  hasOlder: boolean
  loadingOlder: boolean
  userScrolled: boolean
  onLoadOlder: () => void
}

export function OlderMessagesLoader({
  hasOlder,
  loadingOlder,
  userScrolled,
  onLoadOlder,
}: OlderMessagesLoaderProps) {
  const { t } = useTranslation("chat")
  const { start, end } = useMessageScrollerScrollable()
  const shouldLoad = hasOlder && !loadingOlder && !start && (userScrolled || !end)

  useEffect(() => {
    if (shouldLoad) {
      onLoadOlder()
    }
  }, [shouldLoad, onLoadOlder])

  if (!loadingOlder) {
    return null
  }
  return (
    <div
      role="status"
      className="pointer-events-none absolute inset-x-0 top-2 z-10 flex justify-center"
    >
      <span className="flex items-center gap-2 rounded-full bg-background/90 px-3 py-1 text-xs text-muted-foreground shadow-sm">
        <Spinner />
        {t("thread.loadingOlder")}
      </span>
    </div>
  )
}
```

`frontend/app/features/chat/components/AgentUnavailableBanner.tsx`:

```tsx
import { CircleAlertIcon } from "lucide-react"
import { useTranslation } from "react-i18next"

import { Alert, AlertDescription, AlertTitle } from "~/components/ui/alert"

export function AgentUnavailableBanner() {
  const { t } = useTranslation("chat")
  return (
    <Alert className="mx-auto mt-3 w-[calc(100%-2rem)] max-w-3xl">
      <CircleAlertIcon aria-hidden />
      <AlertTitle>{t("agentUnavailable.title")}</AlertTitle>
      <AlertDescription>{t("agentUnavailable.description")}</AlertDescription>
    </Alert>
  )
}
```

- [ ] **Step 5: Write `ChatSession.tsx`**

`frontend/app/features/chat/components/ChatSession.tsx`:

```tsx
import { useChat } from "@ai-sdk/react"
import { useQueryClient } from "@tanstack/react-query"
import { ArrowDownIcon } from "lucide-react"
import { useCallback, useEffect, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { useLocation, useNavigate } from "react-router"

import type { MessageFeedback, MessagePage } from "~/api/gen/schemas"
import { ErrorState } from "~/components/elements/error-state"
import {
  MessageScroller,
  MessageScrollerButton,
  MessageScrollerContent,
  MessageScrollerItem,
  MessageScrollerProvider,
  MessageScrollerViewport,
} from "~/components/ui/message-scroller"
import { toast } from "~/components/ui/toast"
import { type CitationTarget, CitationSheet } from "~/features/citations/CitationSheet"
import {
  invalidateConversations,
  updateConversationTitle,
} from "~/features/conversations/lib/conversation-cache"
import { apiErrorMessage } from "~/i18n/error-message"

import {
  dropTrailingTurn,
  olderMessagesFrom,
  prependOlder,
  questionBefore,
  textOf,
  withFeedback,
} from "../lib/chat-messages"
import { createChatTransport } from "../lib/chat-transport"
import {
  messageMetadataSchema,
  pharmaDataPartSchemas,
  type PharmaUIMessage,
} from "../lib/message-schema"
import type { MessageView } from "../lib/message-view"
import { readStreamProblem } from "../lib/stream-problem"

import { AgentUnavailableBanner } from "./AgentUnavailableBanner"
import { Composer } from "./Composer"
import { MessageActions } from "./MessageActions"
import { MessageParts } from "./MessageParts"
import { OlderMessagesLoader } from "./OlderMessagesLoader"

export type ChatSessionProps = {
  conversationId: string
  initialMessages: PharmaUIMessage[]
  pages: readonly MessagePage[]
  hasOlder: boolean
  loadingOlder: boolean
  onLoadOlder: () => void
  initialQuestion: string | undefined
  onInitialQuestionSent: () => void
}

const STANDALONE_ERROR_ID = "stream-error"

function withId(ids: ReadonlySet<string>, id: string): ReadonlySet<string> {
  const next = new Set(ids)
  next.add(id)
  return next
}

export function ChatSession({
  conversationId,
  initialMessages,
  pages,
  hasOlder,
  loadingOlder,
  onLoadOlder,
  initialQuestion,
  onInitialQuestionSent,
}: ChatSessionProps) {
  const { t } = useTranslation("chat")
  const { t: tErrors } = useTranslation("errors")
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const location = useLocation()
  const [transport] = useState(createChatTransport)
  const [draft, setDraft] = useState("")
  const [composerError, setComposerError] = useState<string | null>(null)
  const [agentUnavailable, setAgentUnavailable] = useState(false)
  const [stoppedIds, setStoppedIds] = useState<ReadonlySet<string>>(() => new Set())
  const [failedIds, setFailedIds] = useState<ReadonlySet<string>>(() => new Set())
  const [retryingId, setRetryingId] = useState<string | null>(null)
  const [citation, setCitation] = useState<CitationTarget | null>(null)
  const [userScrolled, setUserScrolled] = useState(false)
  const lastQuestionRef = useRef("")
  const appliedPagesRef = useRef(1)
  const initialSentRef = useRef(false)

  const { messages, sendMessage, status, stop, setMessages, error } =
    useChat<PharmaUIMessage>({
      id: conversationId,
      messages: initialMessages,
      transport,
      dataPartSchemas: pharmaDataPartSchemas,
      messageMetadataSchema,
      onData: (part) => {
        if (
          part.type === "data-conversation" &&
          !updateConversationTitle(queryClient, part.data)
        ) {
          void invalidateConversations(queryClient)
        }
      },
      onFinish: ({ message, isAbort, isError, isDisconnect }) => {
        setRetryingId(null)
        if (isAbort) {
          setStoppedIds((current) => withId(current, message.id))
        }
        if (isError || isDisconnect) {
          setFailedIds((current) => withId(current, message.id))
        }
        if (message.metadata?.persisted === false) {
          toast.add({ title: t("notPersisted"), type: "warning" })
        }
        void invalidateConversations(queryClient)
      },
      onError: (streamError) => {
        const problem = readStreamProblem(streamError)
        if (problem === null) {
          return
        }
        setMessages((current) => dropTrailingTurn(current))
        setDraft(lastQuestionRef.current)
        if (problem.status === 401) {
          void navigate(
            `/login?next=${encodeURIComponent(location.pathname + location.search)}`
          )
        } else if (problem.code === "CONVERSATION_NOT_FOUND") {
          toast.add({ title: apiErrorMessage(problem, tErrors), type: "error" })
          void navigate("/chat", { replace: true })
        } else if (problem.status === 503) {
          setAgentUnavailable(true)
        } else {
          setComposerError(problem.detail ?? apiErrorMessage(problem, tErrors))
        }
      },
    })

  const busy = status === "submitted" || status === "streaming"
  const lastMessage = messages.at(-1)
  const streamingId =
    busy && lastMessage?.role === "assistant" ? lastMessage.id : undefined
  const streamProblem = error === undefined ? null : readStreamProblem(error)
  const showStandaloneError =
    status === "error" && streamProblem === null && lastMessage?.role === "user"

  const ask = useCallback(
    (question: string) => {
      lastQuestionRef.current = question
      setComposerError(null)
      setAgentUnavailable(false)
      void sendMessage({ text: question })
    },
    [sendMessage]
  )

  const retry = useCallback(
    (messageId: string) => {
      const question = questionBefore(messages, messageId)
      if (question === undefined) {
        return
      }
      setRetryingId(messageId)
      if (failedIds.has(messageId) && messages.at(-1)?.id === messageId) {
        setMessages(dropTrailingTurn(messages))
      }
      ask(question)
    },
    [messages, failedIds, setMessages, ask]
  )

  const retryLastQuestion = useCallback(() => {
    const last = messages.at(-1)
    if (last?.role !== "user") {
      return
    }
    setMessages(messages.slice(0, -1))
    ask(textOf(last))
  }, [messages, setMessages, ask])

  const changeFeedback = useCallback(
    (messageId: string, feedback: MessageFeedback | null) => {
      setMessages((current) => withFeedback(current, messageId, feedback))
    },
    [setMessages]
  )

  const renderActions = useCallback(
    (message: PharmaUIMessage, view: MessageView) => (
      <MessageActions
        conversationId={conversationId}
        messageId={message.id}
        text={view.text}
        feedback={view.feedback}
        onFeedbackChange={changeFeedback}
      />
    ),
    [conversationId, changeFeedback]
  )

  const openCitation = useCallback((messageId: string, index: number) => {
    setCitation({ messageId, index })
  }, [])

  const markUserScrolled = useCallback(() => setUserScrolled(true), [])

  useEffect(() => {
    if (pages.length <= appliedPagesRef.current) {
      return
    }
    const older = olderMessagesFrom(pages, appliedPagesRef.current)
    appliedPagesRef.current = pages.length
    setMessages((current) => prependOlder(current, older))
  }, [pages, setMessages])

  useEffect(() => {
    if (initialQuestion === undefined || initialSentRef.current) {
      return
    }
    initialSentRef.current = true
    lastQuestionRef.current = initialQuestion
    void sendMessage({ text: initialQuestion })
    onInitialQuestionSent()
  }, [initialQuestion, sendMessage, onInitialQuestionSent])

  return (
    <div className="flex h-full min-h-0 flex-col">
      {agentUnavailable ? <AgentUnavailableBanner /> : null}
      <MessageScrollerProvider autoScroll defaultScrollPosition="last-anchor">
        <MessageScroller className="min-h-0 flex-1">
          <OlderMessagesLoader
            hasOlder={hasOlder}
            loadingOlder={loadingOlder}
            userScrolled={userScrolled}
            onLoadOlder={onLoadOlder}
          />
          <MessageScrollerViewport
            aria-label={t("thread.label")}
            onWheel={markUserScrolled}
            onTouchMove={markUserScrolled}
            onKeyDown={markUserScrolled}
          >
            <MessageScrollerContent className="mx-auto w-full max-w-3xl px-4 py-6">
              {messages.map((message) => (
                <MessageScrollerItem
                  key={message.id}
                  messageId={message.id}
                  data-message-id={message.id}
                  scrollAnchor={message.role === "user"}
                >
                  <MessageParts
                    message={message}
                    streaming={message.id === streamingId}
                    stopped={stoppedIds.has(message.id)}
                    failed={failedIds.has(message.id)}
                    retrying={retryingId === message.id}
                    onRetry={retry}
                    onOpenCitation={openCitation}
                    renderActions={renderActions}
                  />
                </MessageScrollerItem>
              ))}
              {showStandaloneError ? (
                <MessageScrollerItem messageId={STANDALONE_ERROR_ID}>
                  <ErrorState
                    title={t("error.title")}
                    detail={t("error.network")}
                    retrying={false}
                    onRetry={retryLastQuestion}
                  />
                </MessageScrollerItem>
              ) : null}
            </MessageScrollerContent>
          </MessageScrollerViewport>
          <MessageScrollerButton direction="end">
            <ArrowDownIcon aria-hidden />
            <span className="sr-only">{t("thread.scrollToEnd")}</span>
          </MessageScrollerButton>
        </MessageScroller>
      </MessageScrollerProvider>
      <Composer
        value={draft}
        onValueChange={(value) => {
          setDraft(value)
          setComposerError(null)
        }}
        busy={busy}
        error={composerError}
        onSubmit={(question) => {
          setDraft("")
          ask(question)
        }}
        onStop={() => void stop()}
      />
      <CitationSheet target={citation} onClose={() => setCitation(null)} />
    </div>
  )
}
```

- [ ] **Step 6: Write `ChatThread.tsx`**

`frontend/app/features/chat/ChatThread.tsx`:

```tsx
import { useCallback, useEffect, useMemo } from "react"
import { useTranslation } from "react-i18next"
import { useLocation, useNavigate } from "react-router"

import { useListMessagesInfinite } from "~/api/gen/endpoints"
import { isApiError } from "~/api/problem"
import { Button } from "~/components/ui/button"
import { Empty, EmptyContent, EmptyHeader, EmptyTitle } from "~/components/ui/empty"
import { Spinner } from "~/components/ui/spinner"
import { toast } from "~/components/ui/toast"
import { apiErrorMessage } from "~/i18n/error-message"

import { ChatSession } from "./components/ChatSession"
import { toPharmaMessage } from "./lib/chat-messages"
import { readInitialQuestion } from "./lib/location-state"

export const HISTORY_PAGE_SIZE = 30

export type ChatThreadProps = {
  conversationId: string | undefined
}

export function ChatThreadFallback() {
  const { t } = useTranslation("chat")
  return (
    <div
      role="status"
      className="flex h-full items-center justify-center gap-2 text-sm text-muted-foreground"
    >
      <Spinner />
      {t("thread.loading")}
    </div>
  )
}

function ConversationChat({ conversationId }: { conversationId: string }) {
  const { t } = useTranslation("chat")
  const { t: tErrors } = useTranslation("errors")
  const navigate = useNavigate()
  const location = useLocation()
  const initialQuestion = readInitialQuestion(location.state)
  const {
    data,
    error,
    isError,
    hasNextPage,
    isFetchingNextPage,
    fetchNextPage,
    refetch,
  } = useListMessagesInfinite(
    conversationId,
    { limit: HISTORY_PAGE_SIZE },
    {
      query: {
        initialPageParam: undefined,
        getNextPageParam: (page) => page.next_cursor ?? undefined,
        staleTime: Number.POSITIVE_INFINITY,
        gcTime: 0,
        refetchOnWindowFocus: false,
        refetchOnReconnect: false,
      },
    }
  )
  const notFound = isApiError(error) && error.code === "CONVERSATION_NOT_FOUND"
  const firstPage = data?.pages[0]
  const initialMessages = useMemo(
    () => firstPage?.items.map(toPharmaMessage) ?? [],
    [firstPage]
  )

  useEffect(() => {
    if (!notFound) {
      return
    }
    toast.add({ title: apiErrorMessage(error, tErrors), type: "error" })
    void navigate("/chat", { replace: true })
  }, [notFound, error, navigate, tErrors])

  const clearInitialQuestion = useCallback(() => {
    void navigate(location.pathname, { replace: true, state: null })
  }, [navigate, location.pathname])

  const loadOlder = useCallback(() => {
    void fetchNextPage()
  }, [fetchNextPage])

  if (isError && data === undefined) {
    if (notFound) {
      return null
    }
    return (
      <Empty className="h-full">
        <EmptyHeader>
          <EmptyTitle>{t("thread.loadFailed")}</EmptyTitle>
        </EmptyHeader>
        <EmptyContent>
          <Button variant="outline" onClick={() => void refetch()}>
            {t("thread.reload")}
          </Button>
        </EmptyContent>
      </Empty>
    )
  }

  if (data === undefined || firstPage === undefined) {
    return <ChatThreadFallback />
  }

  return (
    <ChatSession
      conversationId={conversationId}
      initialMessages={initialMessages}
      pages={data.pages}
      hasOlder={hasNextPage}
      loadingOlder={isFetchingNextPage}
      onLoadOlder={loadOlder}
      initialQuestion={initialQuestion}
      onInitialQuestionSent={clearInitialQuestion}
    />
  )
}

export function ChatThread({ conversationId }: ChatThreadProps) {
  if (conversationId === undefined) {
    return null
  }
  return <ConversationChat key={conversationId} conversationId={conversationId} />
}
```

- [ ] **Step 7: Replace the chat route module**

`frontend/app/routes/app/chat.tsx` (whole file; P8's route table already maps `chat/:conversationId?` here, and P8's layout renders a `h-12` header above the outlet):

```tsx
import { ChatThread, ChatThreadFallback } from "~/features/chat/ChatThread"

import type { Route } from "./+types/chat"

export function clientLoader({ params }: Route.ClientLoaderArgs) {
  return { conversationId: params.conversationId ?? null }
}

export function HydrateFallback() {
  return <ChatThreadFallback />
}

export default function ChatRoute({ loaderData }: Route.ComponentProps) {
  return (
    <div className="flex h-[calc(100svh-3rem)] min-h-0 flex-col">
      <ChatThread conversationId={loaderData.conversationId ?? undefined} />
    </div>
  )
}
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `cd frontend && npx react-router typegen && npx vitest run --project browser app/features/chat/ChatThread.browser.test.tsx app/features/chat/history-prepend.browser.test.tsx`
Expected: PASS (15 thread tests, 1 prepend test).

- [ ] **Step 9: Full check**

Run: `cd frontend && npm run format:write && npm run lint && npm run format && npm run typecheck && npm test`
Expected: all green.

- [ ] **Step 10: Commit**

```bash
git add frontend/app/features/chat/ChatThread.tsx frontend/app/features/chat/ChatThread.browser.test.tsx \
  frontend/app/features/chat/history-prepend.browser.test.tsx \
  frontend/app/features/chat/components/ChatSession.tsx frontend/app/features/chat/components/Composer.tsx \
  frontend/app/features/chat/components/OlderMessagesLoader.tsx frontend/app/features/chat/components/AgentUnavailableBanner.tsx \
  frontend/app/features/chat/lib/location-state.ts frontend/tests/chat/stream-handler.ts \
  frontend/app/routes/app/chat.tsx
git commit -m "feat(chat): stream chat turns with history paging and error handling" \
  -m "$COMMIT_TRAILER"
```

---

### Task 10: New conversation flow (spec B §10.2)

**Files:**
- Create: `frontend/app/features/chat/components/NewChat.tsx`
- Modify: `frontend/app/features/chat/ChatThread.tsx`
- Test: `frontend/app/features/chat/NewChat.browser.test.tsx`

**Interfaces:**
- Consumes: `useCreateConversation` (orval; P5 `create_conversation` has no body, so mutation variables are `void`); `getCreateConversationMockHandler`, `getListMessagesMockHandler` (`~/api/gen/endpoints.msw`); `ChatLocationState`, `Composer` (Task 9); stream test support (Task 9).
- Produces: `NewChat()`; `ChatThread` renders it when `conversationId` is `undefined`.

Flow: on `/chat` the user sends the first question → `POST /conversations` → `navigate("/chat/<id>", { replace: true, state: { initialQuestion } })` → `ChatThread` mounts `ConversationChat` for the new id, loads the empty first history page, and `ChatSession` sends the question once and clears the navigation state (Task 9). The title arrives through `data-conversation`; the list refreshes when the turn finishes. The stream starts only after the conversation exists and `useChat` is created with the final id, so no in-flight stream is stopped by an id change.

- [ ] **Step 1: Write the failing test**

`frontend/app/features/chat/NewChat.browser.test.tsx`:

```tsx
import { http, HttpResponse } from "msw"
import { useParams } from "react-router"
import { describe, expect, test } from "vitest"
import { page, userEvent } from "vitest/browser"

import {
  getCreateConversationMockHandler,
  getListMessagesMockHandler,
} from "~/api/gen/endpoints.msw"

import { conversationView, messagePage } from "../../../tests/chat/fixtures"
import { renderRoutes } from "../../../tests/chat/render"
import {
  answerChat,
  chatStreamHandler,
  type StreamRequest,
} from "../../../tests/chat/stream-handler"
import { worker } from "../../../tests/msw/browser"

import { ChatThread } from "./ChatThread"

const QUESTION = "Ibuprofen có dùng chung với aspirin được không?"

function ChatPage() {
  const { conversationId } = useParams()
  return (
    <div className="h-[640px]">
      <ChatThread conversationId={conversationId} />
    </div>
  )
}

describe("new conversation", () => {
  test("creates the conversation, moves to its URL and sends the first question once", async () => {
    const created: string[] = []
    const requests: StreamRequest[] = []
    worker.use(
      http.post("*/api/v1/conversations", () => {
        created.push("c-new")
        return HttpResponse.json(
          conversationView("c-new", "Cuộc trò chuyện mới", { turn_count: 0 }),
          { status: 201 }
        )
      }),
      getListMessagesMockHandler(messagePage([])),
      chatStreamHandler([answerChat(QUESTION, { conversationId: "c-new" })], requests)
    )

    await renderRoutes(ChatPage, { initialEntry: "/chat" })
    await expect.element(page.getByText("Hỏi về thuốc")).toBeVisible()
    await userEvent.fill(page.getByRole("textbox", { name: "Tin nhắn" }), QUESTION)
    await page.getByRole("button", { name: "Gửi" }).click()

    await expect.element(page.getByTestId("location")).toHaveTextContent(/^\/chat\/c-new$/)
    await expect.element(page.getByText(QUESTION, { exact: true })).toBeVisible()
    await expect.element(page.getByRole("button", { name: "1", exact: true })).toBeVisible()
    expect(created).toEqual(["c-new"])
    expect(requests).toEqual([{ conversation_id: "c-new", message: QUESTION }])
  })

  test("keeps the draft and shows a toast when the conversation cannot be created", async () => {
    worker.use(
      http.post("*/api/v1/conversations", () =>
        HttpResponse.json(
          {
            type: "urn:pharma-agent:problem:agent-unavailable",
            title: "Agent unavailable",
            status: 503,
            code: "AGENT_UNAVAILABLE",
          },
          { status: 503, headers: { "Content-Type": "application/problem+json" } }
        )
      )
    )
    await renderRoutes(ChatPage, { initialEntry: "/chat" })
    await userEvent.fill(page.getByRole("textbox", { name: "Tin nhắn" }), QUESTION)
    await page.getByRole("button", { name: "Gửi" }).click()
    await expect.element(page.getByText("Không tạo được cuộc trò chuyện")).toBeVisible()
    await expect.element(page.getByTestId("location")).toHaveTextContent(/^\/chat$/)
    await expect
      .element(page.getByRole("textbox", { name: "Tin nhắn" }))
      .toHaveValue(QUESTION)
  })

  test("works with the generated create handler and the Enter key", async () => {
    worker.use(
      getCreateConversationMockHandler(
        conversationView("c-generated", "Cuộc trò chuyện mới", { turn_count: 0 })
      ),
      getListMessagesMockHandler(messagePage([])),
      chatStreamHandler([answerChat(QUESTION, { conversationId: "c-generated" })])
    )
    await renderRoutes(ChatPage, { initialEntry: "/chat" })
    await userEvent.fill(page.getByRole("textbox", { name: "Tin nhắn" }), QUESTION)
    await userEvent.keyboard("{Enter}")
    await expect
      .element(page.getByTestId("location"))
      .toHaveTextContent(/^\/chat\/c-generated$/)
  })
})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd frontend && npx vitest run --project browser app/features/chat/NewChat.browser.test.tsx`
Expected: FAIL; `getByText("Hỏi về thuốc")` times out because `ChatThread` renders nothing for `/chat`.

- [ ] **Step 3: Write `NewChat.tsx`**

`frontend/app/features/chat/components/NewChat.tsx`:

```tsx
import { MessageSquareTextIcon } from "lucide-react"
import { useState } from "react"
import { useTranslation } from "react-i18next"
import { useNavigate } from "react-router"

import { useCreateConversation } from "~/api/gen/endpoints"
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "~/components/ui/empty"
import { toast } from "~/components/ui/toast"

import type { ChatLocationState } from "../lib/location-state"

import { Composer } from "./Composer"

export function NewChat() {
  const { t } = useTranslation("chat")
  const navigate = useNavigate()
  const [draft, setDraft] = useState("")
  const createConversation = useCreateConversation()

  function start(question: string) {
    createConversation.mutate(undefined, {
      onSuccess: (conversation) => {
        const state: ChatLocationState = { initialQuestion: question }
        void navigate(`/chat/${conversation.id}`, { replace: true, state })
      },
      onError: () => {
        toast.add({ title: t("newChat.createFailed"), type: "error" })
      },
    })
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <Empty className="flex-1">
        <EmptyHeader>
          <EmptyMedia variant="icon">
            <MessageSquareTextIcon aria-hidden />
          </EmptyMedia>
          <EmptyTitle>{t("newChat.title")}</EmptyTitle>
          <EmptyDescription>{t("newChat.description")}</EmptyDescription>
        </EmptyHeader>
      </Empty>
      <Composer
        value={draft}
        onValueChange={setDraft}
        busy={false}
        disabled={createConversation.isPending}
        error={null}
        onSubmit={start}
        onStop={() => undefined}
      />
    </div>
  )
}
```

- [ ] **Step 4: Render it from `ChatThread`**

In `frontend/app/features/chat/ChatThread.tsx`, add the import after `import { ChatSession } from "./components/ChatSession"`:

```tsx
import { NewChat } from "./components/NewChat"
```

and make `ChatThread` read:

```tsx
export function ChatThread({ conversationId }: ChatThreadProps) {
  if (conversationId === undefined) {
    return <NewChat />
  }
  return <ConversationChat key={conversationId} conversationId={conversationId} />
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd frontend && npx vitest run --project browser app/features/chat/NewChat.browser.test.tsx app/features/chat/ChatThread.browser.test.tsx`
Expected: PASS (3 new-conversation tests; the 15 thread tests still pass).

- [ ] **Step 6: Full check**

Run: `cd frontend && npm run format:write && npm run lint && npm run format && npm run typecheck && npm test`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add frontend/app/features/chat/components/NewChat.tsx frontend/app/features/chat/ChatThread.tsx \
  frontend/app/features/chat/NewChat.browser.test.tsx
git commit -m "feat(chat): start a conversation from the first question" \
  -m "$COMMIT_TRAILER"
```

---

### Task 11: `ConversationSidebar` with paging, rename and delete, mounted in the app shell

**Files:**
- Create: `frontend/app/features/conversations/ConversationSidebar.tsx`, `frontend/app/features/conversations/components/RenameConversationDialog.tsx`, `frontend/app/features/conversations/components/DeleteConversationDialog.tsx`
- Modify: `frontend/app/features/shell/AppSidebar.tsx`, `frontend/app/features/shell/AppSidebar.browser.test.tsx` (both from P8 Task 11)
- Test: `frontend/app/features/conversations/ConversationSidebar.browser.test.tsx`

**Interfaces:**
- Consumes: `useListConversationsInfinite`, `useRenameConversation`, `useDeleteConversation` (orval); `getListConversationsMockHandler` (`~/api/gen/endpoints.msw`); `RenameConversationBody` (orval zod); `replaceConversationInCache`, `removeConversationFromCache` (Task 8); `useInView` (`react-intersection-observer`); shadcn `Sidebar*` (including `SidebarGroupAction`, `SidebarMenuAction`, `SidebarMenuSkeleton` from the same file), `DropdownMenu*`, `Dialog*`, `AlertDialog*`, `Field*`, `Input`, `Button`, `TooltipProvider`; react-hook-form + `zodResolver`; `renderRoutes`, fixtures (Task 1); `worker`.
- Produces: `CONVERSATION_PAGE_SIZE = 20`; `ConversationSidebar()` (a `SidebarGroup`, pinned path); `RenameConversationDialog({ conversation, onClose }: { conversation: ConversationView | null; onClose: () => void })`; `DeleteConversationDialog({ conversation, activeConversationId, onClose }: { conversation: ConversationView | null; activeConversationId: string | undefined; onClose: () => void })`.

The list uses the infinite query key `["infinite", "/api/v1/conversations", { limit: 20 }]`, the key the chat session updates for titles and invalidates after each turn. A sentinel row observed with `useInView` fetches the next page when it scrolls into view. The backend hides conversations with `turn_count = 0`, so a new conversation appears after its first turn. P8's sidebar collapses to icons; the conversation group hides in that mode (`group-data-[collapsible=icon]:hidden`).

- [ ] **Step 1: Write the failing test**

`frontend/app/features/conversations/ConversationSidebar.browser.test.tsx`:

```tsx
import { http, HttpResponse } from "msw"
import type { ReactNode } from "react"
import { beforeEach, describe, expect, test } from "vitest"
import { page, userEvent } from "vitest/browser"

import type { ConversationView } from "~/api/gen/schemas"
import { Sidebar, SidebarContent, SidebarProvider } from "~/components/ui/sidebar"
import { TooltipProvider } from "~/components/ui/tooltip"

import { conversationPage, conversationView } from "../../../tests/chat/fixtures"
import { renderRoutes } from "../../../tests/chat/render"
import { worker } from "../../../tests/msw/browser"

import { ConversationSidebar } from "./ConversationSidebar"

function SidebarWrap({ children }: { children: ReactNode }) {
  return (
    <TooltipProvider>
      <SidebarProvider>
        <Sidebar collapsible="none" className="h-[400px]">
          <SidebarContent>{children}</SidebarContent>
        </Sidebar>
      </SidebarProvider>
    </TooltipProvider>
  )
}

function conversations(from: number, to: number): ConversationView[] {
  return Array.from({ length: to - from + 1 }, (_unused, offset) => {
    const number = from + offset
    return conversationView(`c${number}`, `Hội thoại ${number}`)
  })
}

type ListPage = { cursor: string | null; items: ConversationView[]; next: string | null }

function listHandler(pages: readonly ListPage[]) {
  return http.get("*/api/v1/conversations", ({ request }) => {
    const cursor = new URL(request.url).searchParams.get("cursor")
    const match = pages.find((candidate) => candidate.cursor === cursor)
    return HttpResponse.json(conversationPage(match?.items ?? [], match?.next ?? null))
  })
}

describe("ConversationSidebar", () => {
  beforeEach(async () => {
    await page.viewport(1280, 800)
  })

  test("marks the active conversation and loads the next page at the end of the list", async () => {
    worker.use(
      listHandler([
        { cursor: null, items: conversations(1, 20), next: "cursor-2" },
        { cursor: "cursor-2", items: conversations(21, 25), next: null },
      ])
    )
    await renderRoutes(ConversationSidebar, { initialEntry: "/chat/c2", wrap: SidebarWrap })
    await expect
      .element(page.getByRole("link", { name: "Hội thoại 2", exact: true }))
      .toHaveAttribute("aria-current", "page")
    page.getByRole("link", { name: "Hội thoại 20", exact: true }).element().scrollIntoView()
    await expect
      .element(page.getByRole("link", { name: "Hội thoại 25", exact: true }))
      .toBeInTheDocument()
  })

  test("renames a conversation", async () => {
    worker.use(listHandler([{ cursor: null, items: conversations(1, 3), next: null }]))
    const patched: unknown[] = []
    worker.use(
      http.patch("*/api/v1/conversations/:conversationId", async ({ request, params }) => {
        patched.push(await request.json())
        return HttpResponse.json(
          conversationView(String(params["conversationId"]), "Liều paracetamol")
        )
      })
    )
    await renderRoutes(ConversationSidebar, { initialEntry: "/chat", wrap: SidebarWrap })
    await page.getByRole("button", { name: "Thao tác với Hội thoại 1" }).click()
    await page.getByRole("menuitem", { name: "Đổi tên" }).click()
    const title = page.getByRole("textbox", { name: "Tên" })
    await expect.element(title).toHaveValue("Hội thoại 1")
    await userEvent.fill(title, "Liều paracetamol")
    await page.getByRole("button", { name: "Lưu" }).click()
    await expect.element(page.getByRole("link", { name: "Liều paracetamol" })).toBeVisible()
    expect(patched).toEqual([{ title: "Liều paracetamol" }])
  })

  test("deletes the active conversation and returns to /chat", async () => {
    worker.use(listHandler([{ cursor: null, items: conversations(1, 3), next: null }]))
    const deleted: string[] = []
    worker.use(
      http.delete("*/api/v1/conversations/:conversationId", ({ params }) => {
        deleted.push(String(params["conversationId"]))
        return new HttpResponse(null, { status: 204 })
      })
    )
    await renderRoutes(ConversationSidebar, { initialEntry: "/chat/c2", wrap: SidebarWrap })
    await page.getByRole("button", { name: "Thao tác với Hội thoại 2" }).click()
    await page.getByRole("menuitem", { name: "Xoá" }).click()
    await expect
      .element(page.getByText("“Hội thoại 2” và toàn bộ tin nhắn sẽ bị xoá vĩnh viễn."))
      .toBeVisible()
    await page.getByRole("alertdialog").getByRole("button", { name: "Xoá" }).click()
    await expect
      .element(page.getByRole("link", { name: "Hội thoại 2", exact: true }))
      .not.toBeInTheDocument()
    await expect.element(page.getByTestId("location")).toHaveTextContent(/^\/chat$/)
    expect(deleted).toEqual(["c2"])
  })

  test("shows the empty state", async () => {
    worker.use(listHandler([{ cursor: null, items: [], next: null }]))
    await renderRoutes(ConversationSidebar, { initialEntry: "/chat", wrap: SidebarWrap })
    await expect.element(page.getByText("Chưa có cuộc trò chuyện nào")).toBeVisible()
  })
})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd frontend && npx vitest run --project browser app/features/conversations/ConversationSidebar.browser.test.tsx`
Expected: FAIL with `Failed to resolve import "./ConversationSidebar"`.

- [ ] **Step 3: Write the dialogs**

`frontend/app/features/conversations/components/RenameConversationDialog.tsx`:

```tsx
import { zodResolver } from "@hookform/resolvers/zod"
import { useQueryClient } from "@tanstack/react-query"
import { useEffect, useId } from "react"
import { Controller, useForm } from "react-hook-form"
import { useTranslation } from "react-i18next"
import type { z } from "zod"

import { useRenameConversation } from "~/api/gen/endpoints"
import type { ConversationView } from "~/api/gen/schemas"
import { RenameConversationBody } from "~/api/gen/zod"
import { Button } from "~/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "~/components/ui/dialog"
import { Field, FieldError, FieldLabel } from "~/components/ui/field"
import { Input } from "~/components/ui/input"
import { toast } from "~/components/ui/toast"

import { replaceConversationInCache } from "../lib/conversation-cache"

type RenameValues = z.infer<typeof RenameConversationBody>

export type RenameConversationDialogProps = {
  conversation: ConversationView | null
  onClose: () => void
}

export function RenameConversationDialog({
  conversation,
  onClose,
}: RenameConversationDialogProps) {
  const { t } = useTranslation("chat")
  const inputId = useId()
  const queryClient = useQueryClient()
  const renameConversation = useRenameConversation()
  const form = useForm<RenameValues>({
    resolver: zodResolver(RenameConversationBody),
    defaultValues: { title: "" },
  })

  useEffect(() => {
    if (conversation !== null) {
      form.reset({ title: conversation.title })
    }
  }, [conversation, form])

  function submit(values: RenameValues) {
    if (conversation === null) {
      return
    }
    renameConversation.mutate(
      { conversationId: conversation.id, data: { title: values.title.trim() } },
      {
        onSuccess: (updated) => {
          replaceConversationInCache(queryClient, updated)
          onClose()
        },
        onError: () => {
          toast.add({ title: t("conversations.renameFailed"), type: "error" })
        },
      }
    )
  }

  return (
    <Dialog
      open={conversation !== null}
      onOpenChange={(open) => {
        if (!open) {
          onClose()
        }
      }}
    >
      <DialogContent showCloseButton={false}>
        <form
          noValidate
          className="grid gap-4"
          onSubmit={(event) => void form.handleSubmit(submit)(event)}
        >
          <DialogHeader>
            <DialogTitle>{t("conversations.renameTitle")}</DialogTitle>
          </DialogHeader>
          <Controller
            control={form.control}
            name="title"
            render={({ field, fieldState }) => (
              <Field data-invalid={fieldState.invalid}>
                <FieldLabel htmlFor={inputId}>{t("conversations.titleLabel")}</FieldLabel>
                <Input
                  {...field}
                  id={inputId}
                  autoComplete="off"
                  aria-invalid={fieldState.invalid}
                />
                {fieldState.invalid ? <FieldError errors={[fieldState.error]} /> : null}
              </Field>
            )}
          />
          <DialogFooter>
            <Button type="button" variant="outline" onClick={onClose}>
              {t("conversations.cancel")}
            </Button>
            <Button type="submit" disabled={renameConversation.isPending}>
              {t("conversations.save")}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
```

`frontend/app/features/conversations/components/DeleteConversationDialog.tsx`:

```tsx
import { useQueryClient } from "@tanstack/react-query"
import { useTranslation } from "react-i18next"
import { useNavigate } from "react-router"

import { useDeleteConversation } from "~/api/gen/endpoints"
import type { ConversationView } from "~/api/gen/schemas"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "~/components/ui/alert-dialog"
import { toast } from "~/components/ui/toast"

import { removeConversationFromCache } from "../lib/conversation-cache"

export type DeleteConversationDialogProps = {
  conversation: ConversationView | null
  activeConversationId: string | undefined
  onClose: () => void
}

export function DeleteConversationDialog({
  conversation,
  activeConversationId,
  onClose,
}: DeleteConversationDialogProps) {
  const { t } = useTranslation("chat")
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const deleteConversation = useDeleteConversation()

  function confirm() {
    if (conversation === null) {
      return
    }
    const deletedId = conversation.id
    deleteConversation.mutate(
      { conversationId: deletedId },
      {
        onSuccess: () => {
          removeConversationFromCache(queryClient, deletedId)
          onClose()
          if (deletedId === activeConversationId) {
            void navigate("/chat", { replace: true })
          }
        },
        onError: () => {
          toast.add({ title: t("conversations.deleteFailed"), type: "error" })
        },
      }
    )
  }

  return (
    <AlertDialog
      open={conversation !== null}
      onOpenChange={(open) => {
        if (!open) {
          onClose()
        }
      }}
    >
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>{t("conversations.deleteTitle")}</AlertDialogTitle>
          <AlertDialogDescription>
            {t("conversations.deleteDescription", { title: conversation?.title ?? "" })}
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>{t("conversations.cancel")}</AlertDialogCancel>
          <AlertDialogAction
            variant="destructive"
            disabled={deleteConversation.isPending}
            onClick={confirm}
          >
            {t("conversations.deleteConfirm")}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
```

- [ ] **Step 4: Write `ConversationSidebar.tsx`**

`frontend/app/features/conversations/ConversationSidebar.tsx`:

```tsx
import { MoreHorizontalIcon, PencilIcon, PlusIcon, Trash2Icon } from "lucide-react"
import { useEffect, useState } from "react"
import { useTranslation } from "react-i18next"
import { useInView } from "react-intersection-observer"
import { Link, NavLink, useParams } from "react-router"

import { useListConversationsInfinite } from "~/api/gen/endpoints"
import type { ConversationView } from "~/api/gen/schemas"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "~/components/ui/dropdown-menu"
import {
  SidebarGroup,
  SidebarGroupAction,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuAction,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenuSkeleton,
} from "~/components/ui/sidebar"

import { DeleteConversationDialog } from "./components/DeleteConversationDialog"
import { RenameConversationDialog } from "./components/RenameConversationDialog"

export const CONVERSATION_PAGE_SIZE = 20

type PendingAction = {
  kind: "rename" | "delete"
  conversation: ConversationView
} | null

const SKELETON_KEYS = ["first", "second", "third"] as const

export function ConversationSidebar() {
  const { t } = useTranslation("chat")
  const { conversationId } = useParams()
  const [pending, setPending] = useState<PendingAction>(null)
  const { data, isPending, isError, hasNextPage, isFetchingNextPage, fetchNextPage } =
    useListConversationsInfinite(
      { limit: CONVERSATION_PAGE_SIZE },
      {
        query: {
          initialPageParam: undefined,
          getNextPageParam: (page) => page.next_cursor ?? undefined,
        },
      }
    )
  const { ref: sentinelRef, inView } = useInView({ rootMargin: "120px" })

  useEffect(() => {
    if (inView && hasNextPage && !isFetchingNextPage) {
      void fetchNextPage()
    }
  }, [inView, hasNextPage, isFetchingNextPage, fetchNextPage])

  const conversations = data?.pages.flatMap((page) => page.items) ?? []

  return (
    <SidebarGroup className="group-data-[collapsible=icon]:hidden">
      <SidebarGroupLabel>{t("conversations.title")}</SidebarGroupLabel>
      <SidebarGroupAction render={<Link to="/chat" />} aria-label={t("conversations.new")}>
        <PlusIcon aria-hidden />
      </SidebarGroupAction>
      <SidebarGroupContent>
        <SidebarMenu>
          {isPending
            ? SKELETON_KEYS.map((key) => (
                <SidebarMenuItem key={key}>
                  <SidebarMenuSkeleton />
                </SidebarMenuItem>
              ))
            : null}
          {isError && data === undefined ? (
            <SidebarMenuItem>
              <p role="alert" className="px-2 text-xs text-destructive">
                {t("conversations.loadFailed")}
              </p>
            </SidebarMenuItem>
          ) : null}
          {data !== undefined && conversations.length === 0 ? (
            <SidebarMenuItem>
              <p className="px-2 text-xs text-muted-foreground">
                {t("conversations.empty")}
              </p>
            </SidebarMenuItem>
          ) : null}
          {conversations.map((conversation) => (
            <SidebarMenuItem key={conversation.id}>
              <SidebarMenuButton
                render={<NavLink to={`/chat/${conversation.id}`} />}
                isActive={conversation.id === conversationId}
              >
                <span className="truncate">{conversation.title}</span>
              </SidebarMenuButton>
              <DropdownMenu>
                <DropdownMenuTrigger
                  render={
                    <SidebarMenuAction
                      showOnHover
                      aria-label={t("conversations.more", { title: conversation.title })}
                    />
                  }
                >
                  <MoreHorizontalIcon aria-hidden />
                </DropdownMenuTrigger>
                <DropdownMenuContent side="right" align="start">
                  <DropdownMenuItem
                    onClick={() => setPending({ kind: "rename", conversation })}
                  >
                    <PencilIcon aria-hidden />
                    {t("conversations.rename")}
                  </DropdownMenuItem>
                  <DropdownMenuItem
                    variant="destructive"
                    onClick={() => setPending({ kind: "delete", conversation })}
                  >
                    <Trash2Icon aria-hidden />
                    {t("conversations.delete")}
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </SidebarMenuItem>
          ))}
          {hasNextPage ? (
            <SidebarMenuItem ref={sentinelRef}>
              <SidebarMenuSkeleton />
            </SidebarMenuItem>
          ) : null}
        </SidebarMenu>
      </SidebarGroupContent>
      <RenameConversationDialog
        conversation={pending?.kind === "rename" ? pending.conversation : null}
        onClose={() => setPending(null)}
      />
      <DeleteConversationDialog
        conversation={pending?.kind === "delete" ? pending.conversation : null}
        activeConversationId={conversationId}
        onClose={() => setPending(null)}
      />
    </SidebarGroup>
  )
}
```

- [ ] **Step 5: Mount it in P8's `AppSidebar` and keep P8's shell test self-contained**

In `frontend/app/features/shell/AppSidebar.tsx`:
- add `import { ConversationSidebar } from "~/features/conversations/ConversationSidebar"` below `import { UserMenu } from "~/features/shell/UserMenu"`;
- inside `<SidebarContent>`, add `<ConversationSidebar />` directly after the closing `</SidebarGroup>` of the navigation group, so the content reads:

```tsx
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupContent>
            <SidebarMenu>
              {NAV_ITEMS.map(({ to, label, Icon }) => (
                <SidebarMenuItem key={to}>
                  <SidebarMenuButton
                    render={<Link to={to} />}
                    isActive={pathname === to || pathname.startsWith(`${to}/`)}
                    tooltip={t(label)}
                  >
                    <Icon aria-hidden="true" />
                    <span>{t(label)}</span>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
        <ConversationSidebar />
      </SidebarContent>
```

In `frontend/app/features/shell/AppSidebar.browser.test.tsx` (the shell now requests `/api/v1/conversations`, which MSW must handle):
- add `import { getListConversationsMockHandler } from "~/api/gen/endpoints.msw"` below `import type { UserRead } from "~/api/gen/schemas"`;
- make the `beforeEach` read:

```tsx
  beforeEach(async () => {
    await page.viewport(1280, 800)
    worker.use(getListConversationsMockHandler({ items: [], next_cursor: null }))
  })
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cd frontend && npx vitest run --project browser app/features/conversations/ConversationSidebar.browser.test.tsx app/features/shell`
Expected: PASS (4 sidebar tests; P8's shell tests still pass).

- [ ] **Step 7: Full check**

Run: `cd frontend && npm run format:write && npm run lint && npm run format && npm run typecheck && npm test`
Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add frontend/app/features/conversations/ConversationSidebar.tsx \
  frontend/app/features/conversations/ConversationSidebar.browser.test.tsx \
  frontend/app/features/conversations/components/RenameConversationDialog.tsx \
  frontend/app/features/conversations/components/DeleteConversationDialog.tsx \
  frontend/app/features/shell/AppSidebar.tsx frontend/app/features/shell/AppSidebar.browser.test.tsx
git commit -m "feat(conversations): list, rename and delete conversations in the app sidebar" \
  -m "$COMMIT_TRAILER"
```

---

### Task 12: Contract test on the backend UI stream fixtures (spec B §13, spec A §10)

**Files:**
- Create: `frontend/tests/chat/ui-stream-fixtures.ts`
- Modify: `frontend/vitest.config.ts`
- Test: `frontend/app/features/chat/contract/ui-stream.test.ts`, `frontend/app/features/chat/contract/ui-stream.browser.test.tsx`

**Interfaces:**
- Consumes: P6 fixtures `backend/tests/contract/fixtures/ui-stream/{completed-with-citations,blocked,timeout,persist-failed,no-evidence}.sse`; `parseJsonEventStream`, `uiMessageChunkSchema`, `readUIMessageStream` from `ai`; `pharmaDataPartSchemas`, `messageMetadataSchema`, `pharmaSourceMetadataSchema`, `PharmaUIMessage` (Task 3); `MessageParts`, `describeMessage` (Task 5); `renderWithProviders` (Task 1).
- Produces (`tests/chat/ui-stream-fixtures.ts`): `CONTRACT_SCENARIOS`, `type ContractScenario`, `type ChunkParseResult`, `fixtureText(scenario): string`, `parseFixtureChunks(scenario): Promise<ChunkParseResult[]>`, `validChunks(results): UIMessageChunk[]`, `dataPartMatchesSchema(name: string, data: unknown): boolean`, `fixtureMessage(scenario): Promise<PharmaUIMessage>`.

P6 records the real `/chat/stream` output for five scenarios (regenerated with `UPDATE_CONTRACT_FIXTURES=1 uv run pytest -q tests/contract`):

| Scenario | status | finishReason | persisted | Chunk shape |
| --- | --- | --- | --- | --- |
| `completed-with-citations` | `completed` | `stop` | `true` | start, data-conversation, data-phase, data-skills, data-phase, data-evidence, text, source-document ×2 (`[1]`, `[2]`), finish |
| `blocked` | `blocked` | `stop` | `true` | start, data-conversation, data-phase, text, finish |
| `timeout` | `timeout` | `error` | `true` | start, data-conversation, data-phase, data-evidence, text, finish |
| `persist-failed` | `completed` | `stop` | `false` | as `completed-with-citations` |
| `no-evidence` | `abstained` | `stop` | `true` | start, data-conversation, data-phase, text, finish |

The fixtures are read with Vite `import.meta.glob(..., { query: "?raw", import: "default", eager: true })`, so both Vitest projects load the files the backend test writes. They sit outside `frontend/`, so the browser project's Vite server must allow that directory.

- [ ] **Step 1: Allow the fixture directory in Vitest**

In `frontend/vitest.config.ts` (P8 Task 1), add `import { searchForWorkspaceRoot } from "vite"` below `import { defineConfig } from "vitest/config"`, and add a `server` entry to the object passed to `defineConfig`, next to `resolve`:

```ts
  server: {
    fs: {
      allow: [
        searchForWorkspaceRoot(process.cwd()),
        "../backend/tests/contract/fixtures",
      ],
    },
  },
```

- [ ] **Step 2: Write the fixture loader**

`frontend/tests/chat/ui-stream-fixtures.ts`:

```ts
import {
  parseJsonEventStream,
  readUIMessageStream,
  type UIMessageChunk,
  uiMessageChunkSchema,
} from "ai"

import {
  pharmaDataPartSchemas,
  type PharmaUIMessage,
} from "~/features/chat/lib/message-schema"

const fixtureFiles = import.meta.glob<string>(
  "../../../backend/tests/contract/fixtures/ui-stream/*.sse",
  { query: "?raw", import: "default", eager: true }
)

export const CONTRACT_SCENARIOS = [
  "completed-with-citations",
  "blocked",
  "timeout",
  "persist-failed",
  "no-evidence",
] as const

export type ContractScenario = (typeof CONTRACT_SCENARIOS)[number]

type StreamValue<S> = S extends ReadableStream<infer V> ? V : never

export type ChunkParseResult = StreamValue<
  ReturnType<typeof parseJsonEventStream<UIMessageChunk>>
>

export function fixtureText(scenario: ContractScenario): string {
  const entry = Object.entries(fixtureFiles).find(([path]) =>
    path.endsWith(`/${scenario}.sse`)
  )
  if (entry === undefined) {
    throw new Error(
      `Missing backend contract fixture ${scenario}.sse; run the backend contract tests first`
    )
  }
  return entry[1]
}

function textStream(text: string): ReadableStream<Uint8Array> {
  const bytes = new TextEncoder().encode(text)
  return new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(bytes)
      controller.close()
    },
  })
}

async function readAll<T>(stream: ReadableStream<T>): Promise<T[]> {
  const reader = stream.getReader()
  const values: T[] = []
  let result = await reader.read()
  while (!result.done) {
    values.push(result.value)
    result = await reader.read()
  }
  return values
}

export function parseFixtureChunks(
  scenario: ContractScenario
): Promise<ChunkParseResult[]> {
  return readAll(
    parseJsonEventStream({
      stream: textStream(fixtureText(scenario)),
      schema: uiMessageChunkSchema,
    })
  )
}

export function validChunks(results: readonly ChunkParseResult[]): UIMessageChunk[] {
  return results.flatMap((result) => (result.success ? [result.value] : []))
}

export function dataPartMatchesSchema(name: string, data: unknown): boolean {
  switch (name) {
    case "phase":
      return pharmaDataPartSchemas.phase.safeParse(data).success
    case "skills":
      return pharmaDataPartSchemas.skills.safeParse(data).success
    case "evidence":
      return pharmaDataPartSchemas.evidence.safeParse(data).success
    case "conversation":
      return pharmaDataPartSchemas.conversation.safeParse(data).success
    default:
      return false
  }
}

export async function fixtureMessage(
  scenario: ContractScenario
): Promise<PharmaUIMessage> {
  const chunks = validChunks(await parseFixtureChunks(scenario))
  const stream = new ReadableStream<UIMessageChunk>({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(chunk)
      }
      controller.close()
    },
  })
  let last: PharmaUIMessage | undefined
  for await (const message of readUIMessageStream<PharmaUIMessage>({ stream })) {
    last = message
  }
  if (last === undefined) {
    throw new Error(`Fixture ${scenario}.sse produced no message`)
  }
  return last
}
```

- [ ] **Step 3: Write the tests**

`frontend/app/features/chat/contract/ui-stream.test.ts`:

```ts
import { describe, expect, test } from "vitest"

import {
  CONTRACT_SCENARIOS,
  type ContractScenario,
  dataPartMatchesSchema,
  fixtureText,
  parseFixtureChunks,
  validChunks,
} from "../../../../tests/chat/ui-stream-fixtures"
import { messageMetadataSchema, pharmaSourceMetadataSchema } from "../lib/message-schema"

describe.each(CONTRACT_SCENARIOS)("backend UI stream fixture %s", (scenario) => {
  test("parses every chunk with uiMessageChunkSchema and ends with [DONE]", async () => {
    const results = await parseFixtureChunks(scenario)
    expect(results.length).toBeGreaterThan(0)
    for (const result of results) {
      expect(result.success, JSON.stringify(result.rawValue)).toBe(true)
    }
    const types = validChunks(results).map((chunk) => chunk.type)
    expect(types.slice(0, 2)).toEqual(["start", "data-conversation"])
    expect(types.at(-1)).toBe("finish")
    expect(fixtureText(scenario).trimEnd().endsWith("data: [DONE]")).toBe(true)
  })

  test("matches the pharma data part, source and metadata schemas", async () => {
    for (const chunk of validChunks(await parseFixtureChunks(scenario))) {
      if (chunk.type.startsWith("data-") && "data" in chunk) {
        expect(
          dataPartMatchesSchema(chunk.type.slice("data-".length), chunk.data),
          chunk.type
        ).toBe(true)
      }
      if (chunk.type === "source-document") {
        expect(
          pharmaSourceMetadataSchema.safeParse(chunk.providerMetadata?.["pharma"]).success
        ).toBe(true)
      }
      if (chunk.type === "finish") {
        expect(messageMetadataSchema.safeParse(chunk.messageMetadata).success).toBe(true)
      }
    }
  })
})

describe("scenario specifics", () => {
  async function chunksOf(scenario: ContractScenario) {
    return validChunks(await parseFixtureChunks(scenario))
  }

  test.each([
    ["completed-with-citations", "completed", "stop", true, 2],
    ["blocked", "blocked", "stop", true, 0],
    ["timeout", "timeout", "error", true, 0],
    ["persist-failed", "completed", "stop", false, 2],
    ["no-evidence", "abstained", "stop", true, 0],
  ] as const)(
    "%s finishes with status %s, reason %s, persisted %s and %i sources",
    async (scenario, status, finishReason, persisted, sourceCount) => {
      const chunks = await chunksOf(scenario)
      const finish = chunks.find((chunk) => chunk.type === "finish")
      expect(finish?.type === "finish" ? finish.finishReason : undefined).toBe(finishReason)
      const metadata = messageMetadataSchema.parse(
        finish?.type === "finish" ? finish.messageMetadata : undefined
      )
      expect(metadata.status).toBe(status)
      expect(metadata.persisted).toBe(persisted)
      expect(chunks.filter((chunk) => chunk.type === "source-document")).toHaveLength(
        sourceCount
      )
    }
  )
})
```

`frontend/app/features/chat/contract/ui-stream.browser.test.tsx`:

```tsx
import { describe, expect, test, vi } from "vitest"

import { renderWithProviders } from "../../../../tests/chat/render"
import { CONTRACT_SCENARIOS, fixtureMessage } from "../../../../tests/chat/ui-stream-fixtures"
import { MessageParts } from "../components/MessageParts"
import type { PharmaUIMessage } from "../lib/message-schema"
import { describeMessage } from "../lib/message-view"

const CITE_MARKER = /\[(\d{1,3})\](?!\()/g

function renderFinal(message: PharmaUIMessage) {
  return renderWithProviders(
    <MessageParts
      message={message}
      streaming={false}
      stopped={false}
      failed={false}
      retrying={false}
      onRetry={vi.fn<(messageId: string) => void>()}
      onOpenCitation={vi.fn<(messageId: string, index: number) => void>()}
    />
  )
}

describe.each(CONTRACT_SCENARIOS)("rendering backend fixture %s", (scenario) => {
  test("renders the final assistant message", async () => {
    const message = await fixtureMessage(scenario)
    const screen = await renderFinal(message)
    expect(message.role).toBe("assistant")
    expect(screen.container.textContent).not.toBe("")
  })
})

describe("rendered contract details", () => {
  test("renders one marker per cited source", async () => {
    const message = await fixtureMessage("completed-with-citations")
    const view = describeMessage(message, { streaming: false, failed: false })
    const cited = new Set(
      [...view.text.matchAll(CITE_MARKER)].map((match) => Number(match[1]))
    )
    expect(cited).toEqual(new Set([1, 2]))
    expect(new Set(view.sources.map((source) => source.index))).toEqual(cited)
    const screen = await renderFinal(message)
    for (const index of cited) {
      await expect
        .element(screen.getByRole("button", { name: String(index), exact: true }).first())
        .toBeVisible()
    }
  })

  test("renders the guardrail notice for blocked and the error state for timeout", async () => {
    const blocked = await renderFinal(await fixtureMessage("blocked"))
    await expect.element(blocked.getByText("Không thể trả lời câu hỏi này")).toBeVisible()
    await blocked.unmount()
    const timeout = await renderFinal(await fixtureMessage("timeout"))
    await expect.element(timeout.getByText("Agent phản hồi quá lâu.")).toBeVisible()
  })
})
```

- [ ] **Step 4: Run the tests against the fixtures**

Run: `cd frontend && npx vitest run app/features/chat/contract`
Expected: PASS (10 + 5 unit tests, 5 + 2 browser tests). A failure means the frontend schemas and the backend encoder disagree: fix the side that deviates from spec A §3 and P6; never loosen a schema to make a fixture pass. If the loader throws `Missing backend contract fixture`, run `cd backend && uv run pytest -q tests/contract` first (P6 Task 8).

- [ ] **Step 5: Verify the test fails when the contract breaks**

```bash
cd frontend
sed -i 's/"type":"data-phase"/"type":"data-phaze"/' ../backend/tests/contract/fixtures/ui-stream/blocked.sse
npx vitest run --project unit app/features/chat/contract/ui-stream.test.ts
git -C .. checkout -- backend/tests/contract/fixtures/ui-stream/blocked.sse
git -C .. status --short backend/tests/contract/fixtures
```

Expected: the vitest run FAILS on `matches the pharma data part, source and metadata schemas` for `blocked` (`data-phaze`); after the checkout `git status` shows no change under `backend/tests/contract/fixtures`.

- [ ] **Step 6: Full check**

Run: `cd frontend && npm run format:write && npm run lint && npm run format && npm run typecheck && npm test`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add frontend/vitest.config.ts frontend/tests/chat/ui-stream-fixtures.ts \
  frontend/app/features/chat/contract/ui-stream.test.ts frontend/app/features/chat/contract/ui-stream.browser.test.tsx
git commit -m "test(chat): validate and render the backend UI stream fixtures" \
  -m "$COMMIT_TRAILER"
```

---

## Self-Review

### Spec coverage

| Spec item | Task |
| --- | --- |
| B §10.1 components (`ConversationSidebar`, `ChatThread`, `useChat<PharmaUIMessage>`, `MessageScroller` with `last-anchor` and user anchors, Composer 4000 chars) | 9, 11 |
| B §10.1 `PharmaUIMessage` typed from the OpenAPI models, `dataPartSchemas`, `messageMetadataSchema` | 3 |
| B §10.2 new conversation (`POST /conversations`, replace navigation, send with that id, `data-conversation` title in the list cache) | 8 (cache), 9 (`onData`), 10 |
| B §10.3 transport (`/api/v1/chat/stream`, same-origin, CSRF header, body `{conversation_id, message}`) | 6 |
| B §10.4 older messages (top reached, `fetchNextPage`, prepend, `preserveScrollOnPrepend`, stable ids) | 3, 9 (prepend-keeps-position test) |
| B §10.5 rendering table (phase, skills, evidence, text with `cite-ref`, sources, guardrail, error, not persisted, `isCurrent`) | 4, 5, 9 |
| B §10.6 `CitationSheet` (Sheet/Drawer, detail query, header, strategy, static Streamdown, matched highlight and scroll) | 7 |
| B §10.7 feedback (copy, up, down with dialog, initial state from metadata, optimistic update in `useChat` and the query cache) | 8 |
| B §10.8 error table (401, 404, 422, 503, stop, network) | 9 |
| B §10.9 registry elements copied and edited | 2 |
| B §13 unit rows (cite-ref, prepend, metadata mapping) | 3, 4, 5 |
| B §13 component rows (answer with citation, blocked, timeout, not persisted, 401, prepend keeps position) | 9 |
| B §13 contract row (backend `.sse` fixtures validated with `uiMessageChunkSchema` and rendered) | 12 |

### Reconciliation with P8, P5, P6 and P10

- P8: test support from `tests/msw/browser.ts`, `tests/utils/providers.tsx`, `tests/utils/i18n.ts` (chat helpers added under `tests/chat/`); orval mocks from `~/api/gen/endpoints.msw`; `toast.add`; `ApiError`/`apiErrorMessage` for every problem (stream problems are converted into `ApiError`); the chat placeholder route module is replaced and the route table is untouched; `ConversationSidebar` is mounted in `AppSidebar` and P8's shell test gets a conversations handler; `chat.json`/`citations.json` fill P8's `{}` files with identical vi/en key sets (plurals as `_one` and `_other` in both); `errors.json` is unchanged because P8 already has every code shown; P8's Prettier style and Oxlint rules (typed `vi.fn`, no `as` casts, `react/set-state-in-effect`).
- P5: `create_conversation` has no body (mutation variables `void`); `MessagePage` and `ConversationPage` shapes and page sizes; any 503 shows the unavailable banner.
- P6: data-part names, part ids, `round` only while searching, transient `data-conversation` first after `start`, `MessageMetadata` fields including `usage`, `PharmaSourceMetadata` with `isCurrent`, snake_case `CitationDetail` with `HydrateStrategy`, the five fixture scenarios with their status, finish reason, persisted flag and source count.
- P10 contracts table: accessible names `Tin nhắn`, `Gửi`, `Đang tìm kiếm`, marker buttons named by index, `Trích từ mục`, sheet dialog with strategy labels, `Hữu ích`, `Chưa hữu ích`, `Ghi chú`, `Gửi phản hồi`, `[data-message-id]`.
- Commit steps pass `$COMMIT_TRAILER`; no session URL is written in the plan.

### Placeholder scan

Every code step has complete file content or an exact replacement. The edits to P8 files (`app.css` insertion point, `AppSidebar.tsx`, `AppSidebar.browser.test.tsx`, `vitest.config.ts`) name the exact anchor and the code to add.

### Deviations and decisions to confirm

1. **Data-part zod schemas are hand-written.** P6 documents `PharmaDataParts` under `text/event-stream`, where orval's zod client produces `zod.unknown()` (checked with orval 8.32.0). Types come from the generated models, a two-way compile-time check guards drift, and Task 12 validates the schemas against real fixtures.
2. **`PharmaDataPartTypes` mapped type.** It wraps the generated `PharmaDataParts` interface so it satisfies AI SDK's `Record<string, unknown>` constraint.
3. **Feedback reasons go into `note`** (`"[Sai thông tin, Khó hiểu] ghi chú"`), because the feedback endpoint only accepts `{rating, note}`.
4. **Pre-stream 401 inside the chat** navigates in the router to `/login?next=`. History queries still go through P8's QueryClient redirect.
5. **Upstream demo code removed from the registry elements:** scores and meter, the `range` util, `SwapLabel`, and the `InlineCitation` demo paragraph. Tapping a marker opens the sheet, which replaces a `Popover` fallback for `PreviewCard` on touch.

### Risks

- The meaning of `@shadcn/react` 0.3.1 `useMessageScrollerScrollable().start` follows spec B §10.4 (`false` means the top is reached). The prepend test in Task 9 is the guard; it may load one extra page at mount before the first measurement.
- `@shadcn/helpers` 0.2.0 resolves a scripted turn from the transcript; `chatStreamHandler` passes `chat.get(1)` so each request gets the first scripted turn of the chat served for it.
