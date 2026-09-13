import { describe, expect, test } from "vitest"

import { isApiError } from "~/api/problem"

import { readStreamProblem } from "./stream-problem"

describe("readStreamProblem", () => {
  test("turns a problem+json body carried in the error message into ApiError", () => {
    const problem = readStreamProblem(
      new Error(
        JSON.stringify({
          type: "urn:pharma-agent:problem:agent-unavailable",
          title: "Agent unavailable",
          status: 503,
          code: "AGENT_UNAVAILABLE",
        })
      )
    )
    expect(isApiError(problem)).toBe(true)
    expect(problem?.status).toBe(503)
    expect(problem?.code).toBe("AGENT_UNAVAILABLE")
    expect(problem?.message).toBe("Agent unavailable")
    expect(problem?.detail).toBeUndefined()
  })

  test("keeps the detail of a validation problem", () => {
    const problem = readStreamProblem(
      new Error(
        JSON.stringify({
          type: "urn:pharma-agent:problem:validation-error",
          title: "Validation error",
          status: 422,
          code: "VALIDATION_ERROR",
          detail: "Câu hỏi quá dài",
        })
      )
    )
    expect(problem?.detail).toBe("Câu hỏi quá dài")
  })

  test("returns null for a network error or JSON that is not a problem", () => {
    expect(
      readStreamProblem(new Error("Failed to fetch the chat response."))
    ).toBeNull()
    expect(readStreamProblem(new TypeError("Failed to fetch"))).toBeNull()
    expect(
      readStreamProblem(new Error(JSON.stringify({ message: "x" })))
    ).toBeNull()
  })
})
