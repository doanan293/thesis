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
      .element(
        page.getByText("Trích từ mục Paracetamol › Liều lượng và cách dùng")
      )
      .toBeVisible()
    await expect.element(page.getByText("Trang 812–813")).toBeVisible()
    await expect
      .element(
        page.getByText(
          "Người lớn và trẻ em trên 12 tuổi uống 0,5–1 g mỗi 4–6 giờ."
        )
      )
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
    await userEvent.hover(
      screen.getByRole("button", { name: "1", exact: true })
    )
    await expect
      .element(page.getByText("Nguồn đã có phiên bản mới hơn"))
      .toBeVisible()
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
      withCitations(
        [pharmaSource(1)],
        vi.fn<(index: number) => void>(),
        <Harness />
      )
    )
    await expect.element(screen.getByText(/Đang viết/)).toBeVisible()
    expect(
      screen.getByRole("button", { name: "1", exact: true }).elements()
    ).toHaveLength(0)
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
