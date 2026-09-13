import { z } from "zod"

const problemItemSchema = z.object({
  loc: z.array(z.union([z.string(), z.number()])),
  message: z.string(),
  type: z.string(),
})

export type ProblemItem = z.infer<typeof problemItemSchema>

const problemSchema = z.object({
  title: z.string().optional(),
  detail: z.string().nullish(),
  code: z.string(),
  errors: z.array(problemItemSchema).nullish(),
})

/** Bodies such as {"detail": "LOGIN_BAD_CREDENTIALS"} or {"detail": "Not Found"}. */
const detailSchema = z.object({ detail: z.string() })

const ERROR_CODE_PATTERN = /^[A-Z][A-Z0-9_]*$/

export const UNKNOWN_ERROR_CODE = "UNKNOWN"

export class ApiError extends Error {
  override readonly name = "ApiError"
  readonly status: number
  readonly code: string
  readonly detail: string | undefined
  readonly errors: ProblemItem[]

  constructor(init: {
    status: number
    code: string
    title: string
    detail?: string | undefined
    errors?: ProblemItem[]
  }) {
    super(init.title)
    this.status = init.status
    this.code = init.code
    this.detail = init.detail
    this.errors = init.errors ?? []
  }
}

export function isApiError(value: unknown): value is ApiError {
  return value instanceof ApiError
}

async function readJson(response: Response): Promise<unknown> {
  try {
    return await response.json()
  } catch {
    return undefined
  }
}

export async function toApiError(response: Response): Promise<ApiError> {
  const status = response.status
  const fallbackTitle = response.statusText || `HTTP ${status}`
  const body = await readJson(response)

  const problem = problemSchema.safeParse(body)
  if (problem.success) {
    return new ApiError({
      status,
      code: problem.data.code,
      title: problem.data.title ?? fallbackTitle,
      detail: problem.data.detail ?? undefined,
      errors: problem.data.errors ?? [],
    })
  }

  const detail = detailSchema.safeParse(body)
  if (detail.success) {
    return new ApiError({
      status,
      code: ERROR_CODE_PATTERN.test(detail.data.detail)
        ? detail.data.detail
        : UNKNOWN_ERROR_CODE,
      title: fallbackTitle,
      detail: detail.data.detail,
    })
  }

  return new ApiError({
    status,
    code: UNKNOWN_ERROR_CODE,
    title: fallbackTitle,
  })
}
