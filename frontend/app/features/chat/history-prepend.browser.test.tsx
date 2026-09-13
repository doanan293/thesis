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
      http.get(
        "*/api/v1/conversations/:conversationId/messages",
        async ({ request }) => {
          const cursor = new URL(request.url).searchParams.get("cursor")
          cursors.push(cursor)
          if (cursor === null) {
            return HttpResponse.json(
              messagePage(conversationTurns(15, "new"), "cursor-older")
            )
          }
          await olderGate
          return HttpResponse.json(messagePage(conversationTurns(5, "old")))
        }
      )
    )

    await renderRoutes(ChatPage, { initialEntry: "/chat/c1" })
    await expect
      .element(page.getByText("Câu hỏi new 15", { exact: true }))
      .toBeVisible()
    // A thread taller than the viewport requests nothing older until the user scrolls to the top.
    await new Promise((resolve) => {
      setTimeout(resolve, 300)
    })
    expect(cursors).toEqual([null])

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
      expect(
        Math.abs(anchor.getBoundingClientRect().top - topBefore)
      ).toBeLessThanOrEqual(2)
      expect(viewport.scrollTop).toBeGreaterThan(0)
    })
    expect(messageElement("new-u1")).toBe(anchor)
    expect(cursors.filter((cursor) => cursor === "cursor-older")).toHaveLength(
      1
    )
  })
})
