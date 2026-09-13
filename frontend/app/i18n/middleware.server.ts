import { initReactI18next } from "react-i18next"
import { createI18nextMiddleware } from "remix-i18next"

import {
  FALLBACK_LANGUAGE,
  SUPPORTED_LANGUAGES,
  documentPathname,
  i18nextOptions,
  publicPageLocale,
} from "~/i18n/config"
import { localeCookie } from "~/i18n/locale.server"

export const [i18nextMiddleware, getLocale, getInstance] =
  createI18nextMiddleware({
    detection: {
      supportedLanguages: [...SUPPORTED_LANGUAGES],
      fallbackLanguage: FALLBACK_LANGUAGE,
      cookie: localeCookie,
      order: ["custom", "cookie", "header"],
      // Data requests (`/en.data`) must resolve to the language of the page they load.
      findLocale: ({ request }) =>
        Promise.resolve(
          publicPageLocale(documentPathname(new URL(request.url).pathname))
        ),
    },
    i18next: i18nextOptions,
    plugins: [initReactI18next],
  })
