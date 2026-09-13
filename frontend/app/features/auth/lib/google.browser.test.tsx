import { http, HttpResponse } from "msw"
import { afterEach, describe, expect, test, vi } from "vitest"

import { getOauthGoogleCookieAuthorizeMockHandler } from "~/api/gen/endpoints.msw"
import {
  NEXT_PATH_STORAGE_KEY,
  completeGoogleLogin,
  startGoogleLogin,
} from "~/features/auth/lib/google"

import { worker } from "../../../../tests/msw/browser"

const PROBLEM_HEADERS = { "Content-Type": "application/problem+json" }

describe("Google sign-in", () => {
  afterEach(() => {
    window.sessionStorage.clear()
  })

  test("start stores the next path and opens the authorization URL", async () => {
    worker.use(
      getOauthGoogleCookieAuthorizeMockHandler({
        authorization_url:
          "https://accounts.google.com/o/oauth2/v2/auth?state=s1",
      })
    )
    const assign = vi.fn<(url: string) => void>()

    await startGoogleLogin("/skills", assign)

    expect(assign).toHaveBeenCalledExactlyOnceWith(
      "https://accounts.google.com/o/oauth2/v2/auth?state=s1"
    )
    expect(window.sessionStorage.getItem(NEXT_PATH_STORAGE_KEY)).toBe("/skills")
  })

  test("complete forwards code and state and returns the stored next path", async () => {
    let search = ""
    worker.use(
      http.get("/api/v1/auth/google/callback", ({ request }) => {
        search = new URL(request.url).search
        return new HttpResponse(null, { status: 204 })
      })
    )
    window.sessionStorage.setItem(NEXT_PATH_STORAGE_KEY, "/skills")

    const next = await completeGoogleLogin(
      new URL("http://localhost/auth/google/callback?code=abc&state=xyz")
    )

    expect(next).toBe("/skills")
    const params = new URLSearchParams(search)
    expect(params.get("code")).toBe("abc")
    expect(params.get("state")).toBe("xyz")
    expect(window.sessionStorage.getItem(NEXT_PATH_STORAGE_KEY)).toBeNull()
  })

  test("complete rejects with the problem code when the backend refuses", async () => {
    worker.use(
      http.get("/api/v1/auth/google/callback", () =>
        HttpResponse.json(
          {
            type: "urn:pharma-agent:problem:oauth-invalid-state",
            title: "Invalid state",
            status: 400,
            code: "OAUTH_INVALID_STATE",
          },
          { status: 400, headers: PROBLEM_HEADERS }
        )
      )
    )

    await expect(
      completeGoogleLogin(
        new URL("http://localhost/auth/google/callback?code=abc&state=bad")
      )
    ).rejects.toHaveProperty("code", "OAUTH_INVALID_STATE")
  })
})
