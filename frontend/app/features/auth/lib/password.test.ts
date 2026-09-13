import { describe, expect, test } from "vitest"

import {
  PASSWORD_MIN_LENGTH,
  passwordSchema,
} from "~/features/auth/lib/password"

describe("passwordSchema", () => {
  test("requires at least 8 characters and uses the given message", () => {
    const schema = passwordSchema("Mật khẩu cần ít nhất 8 ký tự.")

    expect(PASSWORD_MIN_LENGTH).toBe(8)
    expect(schema.safeParse("1234567").error?.issues[0]?.message).toBe(
      "Mật khẩu cần ít nhất 8 ký tự."
    )
    expect(schema.safeParse("12345678").success).toBe(true)
  })
})
