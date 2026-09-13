import { describe, expect, test } from "vitest"

import { documentPathname, publicPageLocale } from "~/i18n/config"

describe("publicPageLocale", () => {
  test.each([
    ["/", "vi"],
    ["/en", "en"],
    ["/en/", "en"],
    ["/login", null],
    ["/chat/7f0c", null],
  ] as const)("%s -> %s", (pathname, expected) => {
    expect(publicPageLocale(pathname)).toBe(expected)
  })
})

// React Router single-fetch data requests reach middleware with the raw `.data` URL.
describe("documentPathname", () => {
  test.each([
    ["/_.data", "/"],
    ["/en.data", "/en"],
    ["/en/_.data", "/en/"],
    ["/chat/7f0c.data", "/chat/7f0c"],
    ["/", "/"],
    ["/en/", "/en/"],
    ["/login", "/login"],
  ] as const)("%s -> %s", (pathname, expected) => {
    expect(documentPathname(pathname)).toBe(expected)
  })

  test.each([
    ["/_.data", "vi"],
    ["/en.data", "en"],
    ["/en/_.data", "en"],
  ] as const)(
    "data request %s keeps the page language %s",
    (pathname, expected) => {
      expect(publicPageLocale(documentPathname(pathname))).toBe(expected)
    }
  )
})
