import type { MetaDescriptor } from "react-router"

import type { Language } from "~/i18n/config"

export const LANDING_PATHS: Record<Language, string> = {
  vi: "/",
  en: "/en/",
}

const OPEN_GRAPH_LOCALES: Record<Language, string> = {
  vi: "vi_VN",
  en: "en_US",
}

export type LandingMetaData = {
  locale: Language
  title: string
  description: string
  siteUrl: string
}

export function landingMeta({
  locale,
  title,
  description,
  siteUrl,
}: LandingMetaData): MetaDescriptor[] {
  const absolute = (language: Language) =>
    new URL(LANDING_PATHS[language], siteUrl).toString()

  return [
    { title },
    { name: "description", content: description },
    { property: "og:type", content: "website" },
    { property: "og:title", content: title },
    { property: "og:description", content: description },
    { property: "og:url", content: absolute(locale) },
    { property: "og:locale", content: OPEN_GRAPH_LOCALES[locale] },
    { tagName: "link", rel: "canonical", href: absolute(locale) },
    { tagName: "link", rel: "alternate", hrefLang: "vi", href: absolute("vi") },
    { tagName: "link", rel: "alternate", hrefLang: "en", href: absolute("en") },
    {
      tagName: "link",
      rel: "alternate",
      hrefLang: "x-default",
      href: absolute("vi"),
    },
  ]
}
