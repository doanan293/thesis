import { Theme } from "remix-themes"
import { describe, expect, test } from "vitest"

import { themeSessionResolver } from "~/lib/theme.server"

function requestWithCookie(cookie: string | null): Request {
  return new Request("http://localhost/", {
    headers: cookie === null ? {} : { Cookie: cookie },
  })
}

describe("themeSessionResolver", () => {
  test("has no theme until the user picks one", async () => {
    const session = await themeSessionResolver(requestWithCookie(null))
    expect(session.getTheme()).toBeNull()
  })

  test("round-trips the chosen theme through the signed cookie", async () => {
    const session = await themeSessionResolver(requestWithCookie(null))
    session.setTheme(Theme.DARK)
    const setCookie = await session.commit()

    expect(setCookie).toMatch(/^theme=/)
    const cookiePair = setCookie.split(";")[0] ?? ""
    const next = await themeSessionResolver(requestWithCookie(cookiePair))
    expect(next.getTheme()).toBe(Theme.DARK)
  })

  test("ignores a cookie that was not signed by the server", async () => {
    const session = await themeSessionResolver(
      requestWithCookie("theme=eyJ0aGVtZSI6ImRhcmsifQ%3D%3D")
    )
    expect(session.getTheme()).toBeNull()
  })
})
