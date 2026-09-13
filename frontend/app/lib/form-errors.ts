import type { FieldValues, Path, UseFormSetError } from "react-hook-form"

import { isApiError } from "~/api/problem"

/** 422 items go onto their fields; anything else becomes the form-level `root.server` error. */
export function applyApiError<TFieldValues extends FieldValues>(
  error: unknown,
  setError: UseFormSetError<TFieldValues>,
  fields: readonly NoInfer<Path<TFieldValues>>[],
  fallbackMessage: string
): void {
  if (isApiError(error) && error.status === 422) {
    let placed = false
    for (const item of error.errors) {
      const field = fields.find((name) => name === item.loc.at(-1))
      if (field !== undefined) {
        setError(field, { type: "server", message: item.message })
        placed = true
      }
    }
    if (placed) {
      return
    }
  }
  setError("root.server", { type: "server", message: fallbackMessage })
}
