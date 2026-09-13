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
      messagePage([
        apiUserMessage("u0", "Xin chào"),
        apiAssistantMessage("a0", "Chào bạn."),
      ])
    )
  )
}

async function openThread() {
  const result = await renderRoutes(ChatPage, { initialEntry: "/chat/c1" })
  await expect.element(page.getByText("Chào bạn.")).toBeVisible()
  return result
}

async function ask(question = QUESTION) {
  await userEvent.fill(
    page.getByRole("textbox", { name: "Tin nhắn" }),
    question
  )
  await page.getByRole("button", { name: "Gửi" }).click()
}

const marker = () => page.getByRole("button", { name: "1", exact: true })

describe("ChatThread", () => {
  test("streams phase, answer and sources after the history, then offers feedback", async () => {
    withHistory()
    const requests: StreamRequest[] = []
    worker.use(
      chatStreamHandler([answerChat(QUESTION, { slow: true })], requests)
    )
    await openThread()
    await ask()
    await expect.element(page.getByText("Đang tìm kiếm · vòng 1")).toBeVisible()
    await expect.element(marker()).toBeVisible()
    await expect
      .element(page.getByRole("button", { name: "1 nguồn" }))
      .toBeVisible()
    await expect
      .element(
        page.getByRole("button", { name: "Hữu ích", exact: true }).nth(1)
      )
      .toBeVisible()
    await expect
      .element(page.getByText("Đang tìm kiếm · vòng 1"))
      .not.toBeInTheDocument()
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
    await expect
      .element(page.getByRole("heading", { name: "Nguồn [1]" }))
      .toBeVisible()
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
    await expect
      .element(page.getByText("Không thể trả lời câu hỏi này"))
      .toBeVisible()
    await expect
      .element(page.getByText("Mình không thể hỗ trợ yêu cầu này."))
      .toBeVisible()
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
    await expect
      .element(page.getByText("Agent phản hồi quá lâu."))
      .toBeVisible()
    await page.getByRole("button", { name: "Thử lại" }).click()
    await expect.element(marker()).toBeVisible()
    expect(requests.map((request) => request.message)).toEqual([
      QUESTION,
      QUESTION,
    ])
  })

  test("warns when the turn was not persisted and hides feedback for it", async () => {
    withHistory()
    worker.use(
      chatStreamHandler([
        answerChat(QUESTION, { metadata: { persisted: false } }),
      ])
    )
    await openThread()
    await ask()
    await expect
      .element(page.getByText("Không lưu được lượt này"))
      .toBeVisible()
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
    await expect
      .element(page.getByRole("button", { name: "Gửi" }))
      .toBeVisible()
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
    await expect
      .element(page.getByText("Không tìm thấy cuộc trò chuyện."))
      .toBeVisible()
    await expect
      .element(page.getByTestId("location"))
      .toMatchTextContent(/^\/chat$/)
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
    // The restored draft is the composer's text too, so look for the dropped turn in the thread.
    const thread = document.querySelector(
      '[data-slot="message-scroller-viewport"]'
    )
    expect(thread?.textContent).not.toContain(QUESTION)
  })

  test("blocks a question over the length limit before sending", async () => {
    withHistory()
    await openThread()
    await userEvent.fill(
      page.getByRole("textbox", { name: "Tin nhắn" }),
      "x".repeat(4001)
    )
    await expect
      .element(page.getByText("Câu hỏi dài 4001/4000 ký tự"))
      .toBeVisible()
    await expect
      .element(page.getByRole("button", { name: "Gửi" }))
      .toBeDisabled()
  })

  test("shows the agent unavailable banner on 503", async () => {
    withHistory()
    worker.use(problemHandler(503, "AGENT_UNAVAILABLE"))
    await openThread()
    await ask()
    await expect
      .element(page.getByText("Agent tạm thời không sẵn sàng"))
      .toBeVisible()
    await expect
      .element(page.getByRole("textbox", { name: "Tin nhắn" }))
      .toHaveValue(QUESTION)
  })

  test("offers a retry after a network failure", async () => {
    withHistory()
    worker.use(networkErrorHandler())
    await openThread()
    await ask()
    await expect
      .element(page.getByText("Mất kết nối trong lúc trả lời."))
      .toBeVisible()
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
      initialEntry: {
        pathname: "/chat/c1",
        state: { initialQuestion: QUESTION },
      },
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
          {
            status: 404,
            headers: { "Content-Type": "application/problem+json" },
          }
        )
      )
    )
    await renderRoutes(ChatPage, { initialEntry: "/chat/missing" })
    await expect
      .element(page.getByText("Không tìm thấy cuộc trò chuyện."))
      .toBeVisible()
    await expect
      .element(page.getByTestId("location"))
      .toMatchTextContent(/^\/chat$/)
  })

  test("updates the list title from the transient data-conversation part", async () => {
    withHistory()
    worker.use(
      chatStreamHandler([
        answerChat(QUESTION, {
          conversationId: "c1",
          title: "Liều paracetamol",
        }),
      ])
    )
    const { queryClient } = await openThread()
    const key = getListConversationsInfiniteQueryKey({ limit: 20 })
    queryClient.setQueryData<
      InfiniteData<ConversationPage, string | null | undefined>
    >(key, {
      pages: [
        conversationPage([conversationView("c1", "Cuộc trò chuyện mới")]),
      ],
      pageParams: [undefined],
    })
    await ask()
    await vi.waitFor(() =>
      expect(
        queryClient.getQueryData<
          InfiniteData<ConversationPage, string | null | undefined>
        >(key)?.pages[0]?.items[0]?.title
      ).toBe("Liều paracetamol")
    )
  })
})
