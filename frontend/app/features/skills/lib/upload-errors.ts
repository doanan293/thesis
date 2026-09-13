import type { TFunction } from "i18next"

import { isApiError } from "~/api/problem"
import { apiErrorMessage } from "~/i18n/error-message"

export type UploadError = {
  target: "file" | "root.server"
  message: string
}

/**
 * Where a failed `POST /skills` is shown: 409 (name taken), 413 (too large)
 * and 422 (invalid SKILL.md) belong to the file field, anything else to the
 * form. Messages come from the `errors` namespace by problem code, except 422,
 * which shows the validator's own message.
 */
export function toUploadError(
  error: unknown,
  t: TFunction<"skills">,
  tErrors: TFunction<"errors">
): UploadError {
  if (isApiError(error)) {
    if (error.status === 422) {
      const detail =
        error.errors.length > 0
          ? error.errors.map((item) => item.message).join("; ")
          : (error.detail ?? "")
      return { target: "file", message: t("upload.errors.invalid", { detail }) }
    }
    if (error.status === 409 || error.status === 413) {
      return { target: "file", message: apiErrorMessage(error, tErrors) }
    }
  }
  return { target: "root.server", message: apiErrorMessage(error, tErrors) }
}
