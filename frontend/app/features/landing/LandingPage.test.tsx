import { renderToString } from "react-dom/server"
import { I18nextProvider } from "react-i18next"
import { MemoryRouter } from "react-router"
import { describe, expect, test } from "vitest"

import { LandingPage } from "~/features/landing/LandingPage"
import type { Language } from "~/i18n/config"

import { createTestI18n } from "../../../tests/utils/i18n"

function renderLanding(locale: Language): string {
  return renderToString(
    <I18nextProvider i18n={createTestI18n(locale)}>
      <MemoryRouter>
        <LandingPage locale={locale} />
      </MemoryRouter>
    </I18nextProvider>
  )
}

describe("LandingPage", () => {
  test("renders the Vietnamese page with sign-in links, sources and a switch to English", () => {
    const html = renderLanding("vi")

    expect(html).toContain("Tra cứu thông tin thuốc, có trích dẫn nguồn")
    expect(html).toContain("Dược thư Quốc gia Việt Nam")
    expect(html).toContain('href="/login"')
    expect(html).toContain('href="/register"')
    expect(html).toContain('href="/en/"')
  })

  test("renders the English page with a switch back to Vietnamese", () => {
    const html = renderLanding("en")

    expect(html).toContain("Drug information, with cited sources")
    expect(html).toContain("Vietnamese National Drug Formulary")
    // HTML attribute names are case-insensitive; React 19 serializes the prop as `hrefLang`.
    expect(html).toMatch(/hreflang="vi"/i)
  })
})
