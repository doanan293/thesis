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
    await expect
      .element(screen.getByText("Đang tìm kiếm · vòng 2"))
      .toBeVisible()
    // Vitest 5's toHaveTextContent matches the whole text; the status also holds the label.
    await expect
      .element(screen.getByRole("status"))
      .toMatchTextContent("Đang xử lý")
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
    await expect
      .element(screen.getByText("Không thể trả lời câu hỏi này"))
      .toBeVisible()
    await expect.element(screen.getByText("Nội dung giải thích")).toBeVisible()
  })

  test("RetrievalChunks shows the read count and no score", async () => {
    function Harness() {
      const [open, setOpen] = useState(true)
      return (
        <RetrievalChunks
          chunks={[
            {
              id: "1",
              source: "Paracetamol",
              locator: "Trang 812",
              text: "Người lớn 500 mg",
            },
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
            {
              index: 1,
              source: "Dược thư Quốc gia",
              title: "Paracetamol › Liều dùng",
              stale: false,
            },
            {
              index: 2,
              source: "Dược thư Quốc gia",
              title: "Ibuprofen › Tương tác",
              stale: true,
            },
          ]}
          open={open}
          onOpenChange={setOpen}
          onSelect={onSelect}
        />
      )
    }
    const screen = await renderWithProviders(<Harness />)
    await screen.getByRole("button", { name: "2 nguồn" }).click()
    await expect
      .element(screen.getByText("Nguồn đã có phiên bản mới hơn"))
      .toBeVisible()
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
    await expect
      .element(page.getByText("Người lớn và trẻ em trên 12 tuổi…"))
      .toBeVisible()
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
