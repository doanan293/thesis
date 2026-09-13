import { describe, expect, test } from "vitest"

import { ApiError, isApiError, toApiError } from "~/api/problem"

function jsonResponse(
  status: number,
  body: unknown,
  contentType = "application/problem+json"
): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": contentType },
  })
}

function fieldsOf(error: ApiError) {
  return {
    status: error.status,
    code: error.code,
    message: error.message,
    detail: error.detail,
    errors: error.errors,
  }
}

describe("toApiError", () => {
  test("maps an RFC 9457 problem", async () => {
    const error = await toApiError(
      jsonResponse(404, {
        type: "urn:pharma-agent:problem:conversation-not-found",
        title: "Conversation not found",
        status: 404,
        detail: "No conversation 7f0c",
        code: "CONVERSATION_NOT_FOUND",
      })
    )

    expect(error).toBeInstanceOf(ApiError)
    expect(fieldsOf(error)).toEqual({
      status: 404,
      code: "CONVERSATION_NOT_FOUND",
      message: "Conversation not found",
      detail: "No conversation 7f0c",
      errors: [],
    })
  })

  test("keeps the validation items of a 422 problem", async () => {
    const items = [
      {
        loc: ["body", "message"],
        message: "String should have at most 4000 characters",
        type: "string_too_long",
      },
    ]
    const error = await toApiError(
      jsonResponse(422, {
        type: "urn:pharma-agent:problem:validation-error",
        title: "Validation error",
        status: 422,
        code: "VALIDATION_ERROR",
        errors: items,
      })
    )

    expect(fieldsOf(error)).toEqual({
      status: 422,
      code: "VALIDATION_ERROR",
      message: "Validation error",
      detail: undefined,
      errors: items,
    })
  })

  test("reads a fastapi-users error code from a plain detail body", async () => {
    const error = await toApiError(
      jsonResponse(400, { detail: "LOGIN_BAD_CREDENTIALS" }, "application/json")
    )

    expect(fieldsOf(error)).toEqual({
      status: 400,
      code: "LOGIN_BAD_CREDENTIALS",
      message: "HTTP 400",
      detail: "LOGIN_BAD_CREDENTIALS",
      errors: [],
    })
  })

  test("uses UNKNOWN when a detail is prose instead of a code", async () => {
    const error = await toApiError(
      jsonResponse(404, { detail: "Not Found" }, "application/json")
    )

    expect(fieldsOf(error)).toEqual({
      status: 404,
      code: "UNKNOWN",
      message: "HTTP 404",
      detail: "Not Found",
      errors: [],
    })
  })

  test("uses UNKNOWN when the body is not JSON", async () => {
    const error = await toApiError(
      new Response("<html>Bad Gateway</html>", {
        status: 502,
        statusText: "Bad Gateway",
      })
    )

    expect(fieldsOf(error)).toEqual({
      status: 502,
      code: "UNKNOWN",
      message: "Bad Gateway",
      detail: undefined,
      errors: [],
    })
  })
})

describe("isApiError", () => {
  test("recognises only ApiError instances", () => {
    expect(
      isApiError(
        new ApiError({
          status: 401,
          code: "UNAUTHORIZED",
          title: "Unauthorized",
        })
      )
    ).toBe(true)
    expect(isApiError(new Error("boom"))).toBe(false)
    expect(isApiError({ status: 401, code: "UNAUTHORIZED" })).toBe(false)
  })
})
