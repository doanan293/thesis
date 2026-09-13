import { createCookie } from "react-router"
import { z } from "zod"

import { SUPPORTED_LANGUAGES } from "~/i18n/config"

const ONE_YEAR_SECONDS = 60 * 60 * 24 * 365

export const localeCookie = createCookie("lng", {
  path: "/",
  sameSite: "lax",
  httpOnly: true,
  secure: process.env.NODE_ENV === "production",
  maxAge: ONE_YEAR_SECONDS,
})

const localeForm = z.object({ locale: z.enum(SUPPORTED_LANGUAGES) })

export async function setLocale(request: Request): Promise<Response> {
  const parsed = localeForm.safeParse(
    Object.fromEntries(await request.formData())
  )
  if (!parsed.success) {
    return Response.json({ ok: false }, { status: 400 })
  }
  return Response.json(
    { ok: true },
    {
      headers: {
        "Set-Cookie": await localeCookie.serialize(parsed.data.locale),
      },
    }
  )
}
