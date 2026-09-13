import { z } from "zod"

/** Form validation messages follow the UI language. */
export function applyZodLocale(language: string): void {
  z.config(language === "en" ? z.locales.en() : z.locales.vi())
}
