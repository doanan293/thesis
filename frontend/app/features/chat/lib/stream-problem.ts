import { z } from "zod"

import { ApiError } from "~/api/problem"

const streamProblemSchema = z.object({
  status: z.number().int(),
  code: z.string(),
  title: z.string().optional(),
  detail: z.string().nullish(),
})

export function readStreamProblem(error: Error): ApiError | null {
  let payload: unknown
  try {
    payload = JSON.parse(error.message)
  } catch {
    return null
  }
  const parsed = streamProblemSchema.safeParse(payload)
  if (!parsed.success) {
    return null
  }
  const { status, code, title, detail } = parsed.data
  return new ApiError({
    status,
    code,
    title: title ?? code,
    detail: detail ?? undefined,
  })
}
