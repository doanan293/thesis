import { LandingPage } from "~/features/landing/LandingPage"
import { landingMeta } from "~/features/landing/meta"
import { FALLBACK_LANGUAGE, isLanguage } from "~/i18n/config"
import { getInstance, getLocale } from "~/i18n/middleware.server"

import type { Route } from "./+types/landing"

const DEFAULT_SITE_URL = "http://localhost:8080"

export function loader({ context }: Route.LoaderArgs) {
  const detected = getLocale(context)
  const locale = isLanguage(detected) ? detected : FALLBACK_LANGUAGE
  const t = getInstance(context).getFixedT(locale, "landing")

  return {
    locale,
    title: t("meta.title"),
    description: t("meta.description"),
    siteUrl: process.env.PUBLIC_SITE_URL ?? DEFAULT_SITE_URL,
  }
}

export const meta: Route.MetaFunction = ({ loaderData }) =>
  loaderData === undefined ? [] : landingMeta(loaderData)

export default function Landing({ loaderData }: Route.ComponentProps) {
  return <LandingPage locale={loaderData.locale} />
}
