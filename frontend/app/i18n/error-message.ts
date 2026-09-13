import type { TFunction } from "i18next"

import { isApiError } from "~/api/problem"
import { resources } from "~/i18n/config"

export type ErrorCode = keyof (typeof resources)["vi"]["errors"]

export function isErrorCode(code: string): code is ErrorCode {
  return Object.hasOwn(resources.vi.errors, code)
}

/** Message for a failed request: translated by problem code, else the problem title. */
export function apiErrorMessage(
  error: unknown,
  t: TFunction<"errors">
): string {
  if (isApiError(error)) {
    if (isErrorCode(error.code)) {
      return t(error.code)
    }
    if (error.message !== "") {
      return error.message
    }
  }
  return t("UNKNOWN")
}
