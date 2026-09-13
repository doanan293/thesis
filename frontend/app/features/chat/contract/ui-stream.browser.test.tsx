import { describe, expect, test, vi } from "vitest"

import { renderWithProviders } from "../../../../tests/chat/render"
import {
  CONTRACT_SCENARIOS,
  fixtureMessage,
} from "../../../../tests/chat/ui-stream-fixtures"
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

describe.each(CONTRACT_SCENARIOS)(
  "rendering backend fixture %s",
  (scenario) => {
    test("renders the final assistant message", async () => {
      const message = await fixtureMessage(scenario)
      const screen = await renderFinal(message)
      expect(message.role).toBe("assistant")
      expect(screen.container.textContent).not.toBe("")
    })
  }
)

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
        .element(
          screen
            .getByRole("button", { name: String(index), exact: true })
            .first()
        )
        .toBeVisible()
    }
  })

  test("renders the guardrail notice for blocked and the error state for timeout", async () => {
    const blocked = await renderFinal(await fixtureMessage("blocked"))
    await expect
      .element(blocked.getByText("Không thể trả lời câu hỏi này"))
      .toBeVisible()
    await blocked.unmount()
    const timeout = await renderFinal(await fixtureMessage("timeout"))
    await expect
      .element(timeout.getByText("Agent phản hồi quá lâu."))
      .toBeVisible()
  })
})
