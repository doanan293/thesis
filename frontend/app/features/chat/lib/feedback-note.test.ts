import { describe, expect, test } from "vitest"

import { submitFeedbackBodyNoteMax } from "~/api/gen/zod"

import { composeFeedbackNote } from "./feedback-note"

describe("composeFeedbackNote", () => {
  test("puts selected reasons before the note", () => {
    expect(
      composeFeedbackNote(["Sai thông tin", "Khó hiểu"], "  Thiếu liều trẻ em ")
    ).toBe("[Sai thông tin, Khó hiểu] Thiếu liều trẻ em")
  })

  test("handles reasons only, note only and nothing", () => {
    expect(composeFeedbackNote(["Sai thông tin"], "")).toBe("[Sai thông tin]")
    expect(composeFeedbackNote([], "Thiếu")).toBe("Thiếu")
    expect(composeFeedbackNote([], "   ")).toBe("")
  })

  test("never exceeds the backend limit", () => {
    expect(
      composeFeedbackNote(["Sai thông tin"], "x".repeat(5000))
    ).toHaveLength(submitFeedbackBodyNoteMax)
  })
})
