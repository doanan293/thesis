import { submitFeedbackBodyNoteMax } from "~/api/gen/zod"

export function composeFeedbackNote(
  reasonLabels: readonly string[],
  note: string
): string {
  const trimmed = note.trim()
  const prefix = reasonLabels.length === 0 ? "" : `[${reasonLabels.join(", ")}]`
  let combined = trimmed
  if (prefix !== "") {
    combined = trimmed === "" ? prefix : `${prefix} ${trimmed}`
  }
  return combined.slice(0, submitFeedbackBodyNoteMax)
}
