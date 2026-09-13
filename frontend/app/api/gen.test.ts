import { describe, expect, test } from "vitest"

import {
  getUsersCurrentUserQueryKey,
  getUsersCurrentUserUrl,
} from "~/api/gen/endpoints"
import {
  getPharmaAgentAPIMock,
  getUsersCurrentUserMockHandler,
} from "~/api/gen/endpoints.msw"
import type { UserRead } from "~/api/gen/schemas"
import { AuthCookieLoginBody, RegisterRegisterBody } from "~/api/gen/zod"

describe("generated API client", () => {
  test("query keys are relative API paths, the same on server and client", () => {
    expect(getUsersCurrentUserUrl()).toBe("/api/v1/users/me")
    expect(getUsersCurrentUserQueryKey()).toEqual(["/api/v1/users/me"])
  })

  test("zod schemas validate the auth forms", () => {
    expect(
      AuthCookieLoginBody.safeParse({
        username: "an@example.com",
        password: "correct horse",
      }).success
    ).toBe(true)
    expect(
      RegisterRegisterBody.safeParse({
        email: "not-an-email",
        password: "correct horse",
      }).success
    ).toBe(false)
  })

  test("MSW handlers cover the operations", () => {
    const user: UserRead = {
      id: "0b8f7a52-5d4e-4c7b-9a51-3f7d1c2b9e10",
      email: "an@example.com",
    }
    expect(getUsersCurrentUserMockHandler(user).info.path).toBe(
      "*/api/v1/users/me"
    )
    expect(getPharmaAgentAPIMock().length).toBeGreaterThan(0)
  })
})
