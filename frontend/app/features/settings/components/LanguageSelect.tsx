import { useTranslation } from "react-i18next"
import { useFetcher } from "react-router"

import { Field, FieldDescription, FieldLabel } from "~/components/ui/field"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "~/components/ui/select"
import {
  FALLBACK_LANGUAGE,
  type Language,
  SUPPORTED_LANGUAGES,
  isLanguage,
} from "~/i18n/config"

/** Endonyms: each language is named in itself, so these labels are not translated. */
const LANGUAGE_NAMES: Record<Language, string> = {
  vi: "Tiếng Việt",
  en: "English",
}

const LANGUAGE_ITEMS = SUPPORTED_LANGUAGES.map((code) => ({
  value: code,
  label: LANGUAGE_NAMES[code],
}))

export function LanguageSelect() {
  const { t, i18n } = useTranslation("settings")
  const fetcher = useFetcher()
  const current = isLanguage(i18n.resolvedLanguage)
    ? i18n.resolvedLanguage
    : FALLBACK_LANGUAGE

  return (
    <Field>
      <FieldLabel htmlFor="settings-language">{t("language.label")}</FieldLabel>
      <Select
        items={LANGUAGE_ITEMS}
        value={current}
        onValueChange={(value) => {
          if (!isLanguage(value) || value === current) {
            return
          }
          // P8's locale action stores the choice in the `lng` cookie for SSR and later visits.
          void fetcher.submit(
            { locale: value },
            { method: "post", action: "/actions/locale" }
          )
          void i18n.changeLanguage(value)
        }}
      >
        <SelectTrigger id="settings-language" className="w-48">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {LANGUAGE_ITEMS.map((item) => (
            <SelectItem key={item.value} value={item.value}>
              {item.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <FieldDescription>{t("language.description")}</FieldDescription>
    </Field>
  )
}
