import { createInstance, type i18n } from "i18next"
import { initReactI18next } from "react-i18next"

import { i18nextOptions, type Language } from "~/i18n/config"

export function createTestI18n(language: Language = "vi"): i18n {
  const instance = createInstance()
  void instance
    .use(initReactI18next)
    .init({ ...i18nextOptions, lng: language, initAsync: false })
  return instance
}
