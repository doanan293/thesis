import { afterEach, describe, expect, test } from "vitest"

import { CSRF_COOKIE, CSRF_HEADER, readCsrfToken } from "~/lib/csrf"

function expireCookie(name: string): void {
  document.cookie = `${name}=; path=/; max-age=0`
}

describe("readCsrfToken", () => {
  afterEach(() => {
    expireCookie(CSRF_COOKIE)
    expireCookie("theme")
  })

  test("returns undefined before the backend sets the cookie", () => {
    expect(readCsrfToken()).toBeUndefined()
  })

  test("reads csrftoken among other cookies", () => {
    document.cookie = "theme=dark; path=/"
    document.cookie = "csrftoken=abc123; path=/"
    expect(readCsrfToken()).toBe("abc123")
  })

  test("matches the backend CSRF middleware names", () => {
    expect(CSRF_COOKIE).toBe("csrftoken")
    expect(CSRF_HEADER).toBe("x-csrftoken")
  })
})
