import { http, HttpResponse } from "msw"
import { afterEach, describe, expect, test } from "vitest"

import { getUsersCurrentUserMockHandler } from "~/api/gen/endpoints.msw"
import { queryClient } from "~/api/query-client"
import { requireUser } from "~/features/auth/lib/require-user"

import { worker } from "../../../../tests/msw/browser"

describe("requireUser", () => {
  afterEach(() => {
    queryClient.clear()
  })

  test("returns the signed-in user", async () => {
    worker.use(
      getUsersCurrentUserMockHandler({ id: "u1", email: "an@example.com" })
    )

    await expect(
      requireUser(new Request("http://localhost/chat"))
    ).resolves.toEqual({ id: "u1", email: "an@example.com" })
  })

  test("throws a redirect to /login with the requested path on 401", async () => {
    worker.use(
      http.get("/api/v1/users/me", () =>
        HttpResponse.json(
          {
            type: "urn:pharma-agent:problem:unauthorized",
            title: "Unauthorized",
            status: 401,
            code: "UNAUTHORIZED",
          },
          {
            status: 401,
            headers: { "Content-Type": "application/problem+json" },
          }
        )
      )
    )

    const thrown = await requireUser(
      new Request("http://localhost/chat/7f0c?draft=1")
    ).catch((caught: unknown) => caught)

    expect(thrown).toBeInstanceOf(Response)
    expect(
      thrown instanceof Response
        ? [thrown.status, thrown.headers.get("Location")]
        : []
    ).toEqual([302, "/login?next=%2Fchat%2F7f0c%3Fdraft%3D1"])
  })
})
