import { http, HttpResponse } from "msw"
import { afterEach, describe, expect, test, vi } from "vitest"

import { apiBaseUrl, fetcher } from "~/api/fetcher"
import { ApiError } from "~/api/problem"

import { server } from "../../tests/msw/node"

const API = "http://api.test"

describe("fetcher on the server", () => {
  afterEach(() => {
    vi.unstubAllEnvs()
  })

  test("defaults to the backend on localhost", () => {
    vi.stubEnv("API_INTERNAL_URL", undefined)
    expect(apiBaseUrl()).toBe("http://127.0.0.1:8000")
  })

  test("prefixes API_INTERNAL_URL and forwards the caller's cookie", async () => {
    vi.stubEnv("API_INTERNAL_URL", API)
    server.use(
      http.get(`${API}/api/v1/users/me`, ({ request }) =>
        HttpResponse.json({ cookie: request.headers.get("cookie") })
      )
    )

    const body = await fetcher<{ cookie: string | null }>("/api/v1/users/me", {
      headers: { Cookie: "pharma_session=s1" },
    })

    expect(body).toEqual({ cookie: "pharma_session=s1" })
  })

  test("throws ApiError built from problem+json", async () => {
    vi.stubEnv("API_INTERNAL_URL", API)
    server.use(
      http.get(`${API}/api/v1/conversations/7f0c`, () =>
        HttpResponse.json(
          {
            type: "urn:pharma-agent:problem:conversation-not-found",
            title: "Conversation not found",
            status: 404,
            code: "CONVERSATION_NOT_FOUND",
          },
          {
            status: 404,
            headers: { "Content-Type": "application/problem+json" },
          }
        )
      )
    )

    const request = fetcher("/api/v1/conversations/7f0c")

    await expect(request).rejects.toBeInstanceOf(ApiError)
    await expect(request).rejects.toHaveProperty("status", 404)
    await expect(request).rejects.toHaveProperty(
      "code",
      "CONVERSATION_NOT_FOUND"
    )
  })

  test("resolves null for an empty 204 response", async () => {
    vi.stubEnv("API_INTERNAL_URL", API)
    server.use(
      http.post(
        `${API}/api/v1/auth/cookie/logout`,
        () => new HttpResponse(null, { status: 204 })
      )
    )

    await expect(
      fetcher("/api/v1/auth/cookie/logout", { method: "POST" })
    ).resolves.toBeNull()
  })

  test("sends no CSRF header because there is no browser cookie jar", async () => {
    vi.stubEnv("API_INTERNAL_URL", API)
    server.use(
      http.post(`${API}/api/v1/conversations`, ({ request }) =>
        HttpResponse.json(
          { csrf: request.headers.get("x-csrftoken") },
          { status: 201 }
        )
      )
    )

    await expect(
      fetcher("/api/v1/conversations", { method: "POST" })
    ).resolves.toEqual({ csrf: null })
  })
})
