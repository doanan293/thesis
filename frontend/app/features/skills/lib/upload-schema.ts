import { z } from "zod"

import { UploadSkillBody } from "~/api/gen/zod"

export const SKILL_FILE_NAME = "SKILL.md"
/** Same limit as the backend `MAX_SKILL_BYTES`. */
export const MAX_SKILL_BYTES = 64 * 1024

export type UploadSkillMessages = {
  required: string
  wrongName: string
  tooLarge: string
}

/** Narrows the generated multipart body (`file: Blob`) to a named SKILL.md within the limit. */
export function createUploadSkillSchema(messages: UploadSkillMessages) {
  return UploadSkillBody.extend({
    file: z
      .instanceof(File, { error: messages.required })
      .refine((file) => file.name === SKILL_FILE_NAME, {
        error: messages.wrongName,
      })
      .refine((file) => file.size <= MAX_SKILL_BYTES, {
        error: messages.tooLarge,
      }),
  })
}

export type UploadSkillValues = z.infer<
  ReturnType<typeof createUploadSkillSchema>
>
