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
      chatStreamHandler(
        [answerChat(QUESTION, { conversationId: "c-new" })],
        requests
      )
    )

    await renderRoutes(ChatPage, { initialEntry: "/chat" })
    await expect.element(page.getByText("Hỏi về thuốc")).toBeVisible()
    await userEvent.fill(
      page.getByRole("textbox", { name: "Tin nhắn" }),
      QUESTION
    )
    await page.getByRole("button", { name: "Gửi" }).click()

    await expect
      .element(page.getByTestId("location"))
      .toMatchTextContent(/^\/chat\/c-new$/)
    await expect
      .element(page.getByText(QUESTION, { exact: true }))
      .toBeVisible()
    await expect
      .element(page.getByRole("button", { name: "1", exact: true }))
      .toBeVisible()
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
          {
            status: 503,
            headers: { "Content-Type": "application/problem+json" },
          }
        )
      )
    )
    await renderRoutes(ChatPage, { initialEntry: "/chat" })
    await userEvent.fill(
      page.getByRole("textbox", { name: "Tin nhắn" }),
      QUESTION
    )
    await page.getByRole("button", { name: "Gửi" }).click()
    await expect
      .element(page.getByText("Không tạo được cuộc trò chuyện"))
      .toBeVisible()
    await expect
      .element(page.getByTestId("location"))
      .toMatchTextContent(/^\/chat$/)
    await expect
      .element(page.getByRole("textbox", { name: "Tin nhắn" }))
      .toHaveValue(QUESTION)
  })

  test("works with the generated create handler and the Enter key", async () => {
    worker.use(
      getCreateConversationMockHandler(
        conversationView("c-generated", "Cuộc trò chuyện mới", {
          turn_count: 0,
        })
      ),
      getListMessagesMockHandler(messagePage([])),
      chatStreamHandler([
        answerChat(QUESTION, { conversationId: "c-generated" }),
      ])
    )
    await renderRoutes(ChatPage, { initialEntry: "/chat" })
    await userEvent.fill(
      page.getByRole("textbox", { name: "Tin nhắn" }),
      QUESTION
    )
    await userEvent.keyboard("{Enter}")
    await expect
      .element(page.getByTestId("location"))
      .toMatchTextContent(/^\/chat\/c-generated$/)
  })
})
