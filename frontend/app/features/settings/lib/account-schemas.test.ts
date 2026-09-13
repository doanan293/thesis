import { describe, expect, test } from "vitest"

import { createPasswordSchema, createProfileSchema } from "./account-schemas"

const profile = createProfileSchema({
  required: "required",
  tooLong: "too-long",
})
const password = createPasswordSchema({
  tooShort: "too-short",
  tooLong: "too-long",
  mismatch: "mismatch",
})

function messagesOf(result: { error?: { issues: { message: string }[] } }) {
  return result.error?.issues.map((issue) => issue.message) ?? []
}

describe("createProfileSchema", () => {
  test("trims the display name", () => {
    expect(profile.parse({ display_name: "  Bác sĩ An  " })).toEqual({
      display_name: "Bác sĩ An",
    })
  })

  test("rejects blank and overlong names", () => {
    expect(messagesOf(profile.safeParse({ display_name: "   " }))).toEqual([
      "required",
    ])
    expect(
      messagesOf(profile.safeParse({ display_name: "a".repeat(101) }))
    ).toEqual(["too-long"])
  })
})

describe("createPasswordSchema", () => {
  test("accepts matching passwords of valid length", () => {
    expect(
      password.safeParse({ password: "dai-hon-tam", confirm: "dai-hon-tam" })
        .success
    ).toBe(true)
  })

  test("rejects passwords shorter than 8 characters", () => {
    expect(
      messagesOf(
        password.safeParse({ password: "1234567", confirm: "1234567" })
      )
    ).toEqual(["too-short"])
  })

  test("reports a mismatch on the confirm field", () => {
    const issues =
      password.safeParse({ password: "dai-hon-tam", confirm: "khac-hon-tam" })
        .error?.issues ?? []
    expect(
      issues.map((issue) => ({ message: issue.message, path: issue.path }))
    ).toEqual([{ message: "mismatch", path: ["confirm"] }])
  })
})
