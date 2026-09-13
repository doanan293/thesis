import { createInstance } from "i18next"
import { StrictMode, startTransition } from "react"
import { hydrateRoot } from "react-dom/client"
import { I18nextProvider, initReactI18next } from "react-i18next"
import { HydratedRouter } from "react-router/dom"

import { FALLBACK_LANGUAGE, i18nextOptions } from "~/i18n/config"
import { applyZodLocale } from "~/i18n/zod-locale"

async function hydrate() {
  // One explicit instance, handed to the provider, like the server side does per request.
  const i18n = createInstance()
  await i18n.use(initReactI18next).init({
    ...i18nextOptions,
    lng: document.documentElement.lang || FALLBACK_LANGUAGE,
  })
  applyZodLocale(i18n.language)
  i18n.on("languageChanged", applyZodLocale)

  startTransition(() => {
    hydrateRoot(
      document,
      <I18nextProvider i18n={i18n}>
        <StrictMode>
          <HydratedRouter />
        </StrictMode>
      </I18nextProvider>
    )
  })
}

hydrate().catch((error: unknown) => {
  console.error(error)
})
