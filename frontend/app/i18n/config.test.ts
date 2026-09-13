import { describe, expect, test } from "vitest"

import { isLanguage, localeFromPath } from "~/i18n/config"

describe("localeFromPath", () => {
  test.each([
    ["/en", "en"],
    ["/en/", "en"],
    ["/en/features", "en"],
    ["/", null],
    ["/vi", null],
    ["/chat", null],
    ["/english", null],
  ] as const)("%s -> %s", (pathname, expected) => {
    expect(localeFromPath(pathname)).toBe(expected)
  })
})

describe("isLanguage", () => {
  test("accepts only supported languages", () => {
    expect(isLanguage("vi")).toBe(true)
    expect(isLanguage("en")).toBe(true)
    expect(isLanguage("fr")).toBe(false)
    expect(isLanguage(undefined)).toBe(false)
  })
})
