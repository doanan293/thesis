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
          {
            type: "data-phase",
            id: "phase",
            data: { phase: "searching", round: 2 },
          },
          {
            type: "data-skills",
            data: {
              skills: [{ name: "drug-monograph", title: "Chuyên luận thuốc" }],
            },
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
    await expect
      .element(screen.getByText("Đang tìm kiếm · vòng 2"))
      .toBeVisible()
    await expect.element(screen.getByText("Chuyên luận thuốc")).toBeVisible()
    await expect.element(screen.getByText("Đang tìm tài liệu")).toBeVisible()
    await expect
      .element(screen.getByText("Người lớn uống 500 mg"))
      .toBeVisible()
    await expect.element(screen.getByText("Trang 812")).toBeVisible()
  })

  test("renders a completed answer with sources that open citations", async () => {
    const { screen, props } = await renderParts({
      message: toPharmaMessage(
        apiAssistantMessage(
          "a1",
          "Người lớn 500 mg [1], trẻ em theo cân nặng [2].",
          [
            pharmaSource(1),
            pharmaSource(2, {
              title: "Ibuprofen",
              section: "Tương tác",
              isCurrent: false,
            }),
          ]
        )
      ),
    })
    await expect.element(screen.getByText(/Người lớn 500 mg/)).toBeVisible()
    await screen.getByRole("button", { name: "2 nguồn" }).click()
    await screen.getByRole("button", { name: /Ibuprofen › Tương tác/ }).click()
    expect(props.onOpenCitation).toHaveBeenCalledWith("a1", 2)
    expect(
      screen.container.querySelector("[data-slot=agent-status]")
    ).toBeNull()
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
    await expect
      .element(screen.getByText("Mình chỉ hỗ trợ tra cứu thuốc."))
      .toBeVisible()
  })

  test("shows a timeout with retry", async () => {
    const { screen, props } = await renderParts({
      message: toPharmaMessage(
        apiAssistantMessage("a1", "", [], { status: "timeout" })
      ),
    })
    await expect
      .element(screen.getByText("Agent phản hồi quá lâu."))
      .toBeVisible()
    await screen.getByRole("button", { name: "Thử lại" }).click()
    expect(props.onRetry).toHaveBeenCalledWith("a1")
  })

  test("marks a stopped message", async () => {
    const { screen } = await renderParts({
      stopped: true,
      message: {
        id: "a1",
        role: "assistant",
        parts: [{ type: "text", text: "Người lớn" }],
      },
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
