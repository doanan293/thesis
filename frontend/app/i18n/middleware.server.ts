import { initReactI18next } from "react-i18next"
import { createI18nextMiddleware } from "remix-i18next"

import {
  FALLBACK_LANGUAGE,
  SUPPORTED_LANGUAGES,
  i18nextOptions,
  localeFromPath,
} from "~/i18n/config"
import { localeCookie } from "~/i18n/locale.server"

export const [i18nextMiddleware, getLocale, getInstance] =
  createI18nextMiddleware({
    detection: {
      supportedLanguages: [...SUPPORTED_LANGUAGES],
      fallbackLanguage: FALLBACK_LANGUAGE,
      cookie: localeCookie,
      order: ["custom", "cookie", "header"],
      findLocale: ({ request }) =>
        Promise.resolve(localeFromPath(new URL(request.url).pathname)),
    },
    i18next: i18nextOptions,
    plugins: [initReactI18next],
  })
