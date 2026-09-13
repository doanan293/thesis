import type { UseFormSetError } from "react-hook-form"
import { describe, expect, test, vi } from "vitest"

import { ApiError } from "~/api/problem"
import { applyApiError } from "~/lib/form-errors"

type Values = { email: string; password: string }

const FIELDS = ["email", "password"] as const

describe("applyApiError", () => {
  test("puts 422 validation items on their fields", () => {
    const setError = vi.fn<UseFormSetError<Values>>()
    const error = new ApiError({
      status: 422,
      code: "VALIDATION_ERROR",
      title: "Validation error",
      errors: [
        {
          loc: ["body", "password"],
          message: "String should have at least 8 characters",
          type: "string_too_short",
        },
      ],
    })

    applyApiError<Values>(error, setError, FIELDS, "fallback")

    expect(setError).toHaveBeenCalledExactlyOnceWith("password", {
      type: "server",
      message: "String should have at least 8 characters",
    })
  })

  test("uses root.server when no field matches", () => {
    const setError = vi.fn<UseFormSetError<Values>>()
    const error = new ApiError({
      status: 400,
      code: "REGISTER_USER_ALREADY_EXISTS",
      title: "User already exists",
    })

    applyApiError<Values>(error, setError, FIELDS, "Email này đã được đăng ký.")

    expect(setError).toHaveBeenCalledExactlyOnceWith("root.server", {
      type: "server",
      message: "Email này đã được đăng ký.",
    })
  })
})
