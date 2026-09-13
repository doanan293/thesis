import { z } from "zod"

import { UsersPatchCurrentUserBody } from "~/api/gen/zod"

export const DISPLAY_NAME_MAX_LENGTH = 100
/** Overview §4: passwords entered in the UI need at least 8 characters (as in P8's register form). */
export const PASSWORD_MIN_LENGTH = 8
export const PASSWORD_MAX_LENGTH = 128

export type ProfileMessages = { required: string; tooLong: string }
export type PasswordMessages = {
  tooShort: string
  tooLong: string
  mismatch: string
}

/** `PATCH /users/me` body restricted to a required `display_name`. */
export function createProfileSchema(messages: ProfileMessages) {
  return UsersPatchCurrentUserBody.pick({ display_name: true }).extend({
    display_name: z
      .string()
      .trim()
      .min(1, { error: messages.required })
      .max(DISPLAY_NAME_MAX_LENGTH, { error: messages.tooLong }),
  })
}

/** `PATCH /users/me` body restricted to `password`, plus a confirmation that is never sent. */
export function createPasswordSchema(messages: PasswordMessages) {
  return UsersPatchCurrentUserBody.pick({ password: true })
    .extend({
      password: z
        .string()
        .min(PASSWORD_MIN_LENGTH, { error: messages.tooShort })
        .max(PASSWORD_MAX_LENGTH, { error: messages.tooLong }),
      confirm: z.string(),
    })
    .refine((values) => values.password === values.confirm, {
      error: messages.mismatch,
      path: ["confirm"],
    })
}
