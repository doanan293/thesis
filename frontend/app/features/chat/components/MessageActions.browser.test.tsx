import { http, HttpResponse } from "msw"
import { useState } from "react"
import { describe, expect, test, vi } from "vitest"
import { page, userEvent } from "vitest/browser"

import type { MessageFeedback } from "~/api/gen/schemas"

import { renderRoutes } from "../../../../tests/chat/render"
import { worker } from "../../../../tests/msw/browser"

import { MessageActions } from "./MessageActions"

const FEEDBACK_URL = "*/api/v1/messages/:messageId/feedback"

type FeedbackChange = (
  messageId: string,
  feedback: MessageFeedback | null
) => void

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
    await renderRoutes(actionsPage(vi.fn<FeedbackChange>()), {
      initialEntry: "/chat/c1",
    })
    await page.getByRole("button", { name: "Chưa hữu ích" }).click()
    await page.getByRole("button", { name: "Sai thông tin" }).click()
    await userEvent.fill(
      page.getByRole("textbox", { name: "Ghi chú" }),
      "Thiếu liều trẻ em"
    )
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
          {
            status: 404,
            headers: { "Content-Type": "application/problem+json" },
          }
        )
      )
    )
    const onChange = vi.fn<FeedbackChange>()
    await renderRoutes(actionsPage(onChange), { initialEntry: "/chat/c1" })
    const up = page.getByRole("button", { name: "Hữu ích", exact: true })
    await up.click()
    await expect
      .element(page.getByText("Không gửi được đánh giá"))
      .toBeVisible()
    await expect.element(up).toHaveAttribute("aria-pressed", "false")
    expect(onChange).toHaveBeenLastCalledWith("a1", null)
  })

  test("copies the answer text", async () => {
    const writeText = vi
      .spyOn(navigator.clipboard, "writeText")
      .mockResolvedValue(undefined)
    await renderRoutes(actionsPage(vi.fn<FeedbackChange>()), {
      initialEntry: "/chat/c1",
    })
    await page.getByRole("button", { name: "Sao chép" }).click()
    expect(writeText).toHaveBeenCalledWith("Người lớn 500 mg [1].")
    await expect
      .element(page.getByRole("button", { name: "Đã sao chép" }))
      .toBeVisible()
  })
})
