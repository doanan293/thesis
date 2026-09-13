import { describe, expect, test } from "vitest"

import { ApiError } from "~/api/problem"
import { apiErrorMessage, isErrorCode } from "~/i18n/error-message"

import { createTestI18n } from "../../tests/utils/i18n"

const t = createTestI18n("vi").getFixedT("vi", "errors")

describe("apiErrorMessage", () => {
  test("translates a known problem code", () => {
    const error = new ApiError({
      status: 400,
      code: "LOGIN_BAD_CREDENTIALS",
      title: "Bad credentials",
    })
    expect(apiErrorMessage(error, t)).toBe("Email hoặc mật khẩu không đúng.")
  })

  test("falls back to the problem title for an untranslated code", () => {
    const error = new ApiError({
      status: 409,
      code: "SOMETHING_NEW",
      title: "Something new happened",
    })
    expect(apiErrorMessage(error, t)).toBe("Something new happened")
  })

  test("uses the generic message for errors that are not ApiError", () => {
    expect(apiErrorMessage(new TypeError("Failed to fetch"), t)).toBe(
      "Đã xảy ra lỗi. Vui lòng thử lại."
    )
  })
})

describe("isErrorCode", () => {
  test("knows the codes of the errors namespace only", () => {
    expect(isErrorCode("AGENT_UNAVAILABLE")).toBe(true)
    expect(isErrorCode("toString")).toBe(false)
    expect(isErrorCode("SOMETHING_NEW")).toBe(false)
  })
})
