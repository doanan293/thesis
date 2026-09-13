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
    return (
      <CitationSheet target={{ messageId: "a1", index: 1 }} onClose={onClose} />
    )
  }
}

describe("CitationSheet", () => {
  afterEach(async () => {
    await page.viewport(1280, 800)
  })

  test("shows the source header, strategy and the highlighted matched chunk", async () => {
    await page.viewport(1280, 800)
    worker.use(
      getGetMessageCitationMockHandler(citationDetail(1, { is_current: false }))
    )
    await renderRoutes(sheetPage(vi.fn<() => void>()), {
      initialEntry: "/chat/c1",
    })

    await expect.element(page.getByRole("dialog")).toBeVisible()
    await expect
      .element(page.getByRole("heading", { name: "Nguồn [1]" }))
      .toBeVisible()
    await expect
      .element(page.getByText("Paracetamol › Liều lượng và cách dùng"))
      .toBeVisible()
    await expect.element(page.getByText("Toàn bộ mục")).toBeVisible()
    await expect.element(page.getByText("Trang 811–812")).toBeVisible()
    await expect
      .element(page.getByText("Nguồn đã có phiên bản mới hơn"))
      .toBeVisible()
    await expect.element(page.getByText("Liều thường dùng")).toBeVisible()
    const matched = page.getByRole("region", { name: "Đoạn được trích dẫn" })
    await expect
      .element(matched)
      .toMatchTextContent("Người lớn và trẻ em trên 12 tuổi")
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
          {
            status: 404,
            headers: { "Content-Type": "application/problem+json" },
          }
        )
      )
    )
    await renderRoutes(sheetPage(vi.fn<() => void>()), {
      initialEntry: "/chat/c1",
    })
    await expect
      .element(page.getByRole("alert"))
      .toMatchTextContent("Không tìm thấy nguồn trích dẫn")
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
      getGetMessageCitationMockHandler(
        citationDetail(1, { strategy: "chunk_window" })
      )
    )
    await renderRoutes(sheetPage(vi.fn<() => void>()), {
      initialEntry: "/chat/c1",
    })
    await expect
      .element(page.getByText("Đoạn và các đoạn lân cận"))
      .toBeVisible()
    expect(document.querySelector('[data-slot="sheet-content"]')).toBeNull()
  })
})
