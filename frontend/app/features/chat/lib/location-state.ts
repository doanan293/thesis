import { z } from "zod"

const chatLocationStateSchema = z.object({ initialQuestion: z.string().min(1) })

export type ChatLocationState = z.infer<typeof chatLocationStateSchema>

export function readInitialQuestion(state: unknown): string | undefined {
  const parsed = chatLocationStateSchema.safeParse(state)
  return parsed.success ? parsed.data.initialQuestion : undefined
}
