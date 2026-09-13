import { renderToString } from "react-dom/server"
import { I18nextProvider } from "react-i18next"
import { Theme, ThemeProvider } from "remix-themes"
import { describe, expect, test } from "vitest"

import { Document } from "~/components/document"
import type { Language } from "~/i18n/config"

import { createTestI18n } from "../../tests/utils/i18n"

function renderDocument(language: Language, theme: Theme | null): string {
  return renderToString(
    <I18nextProvider i18n={createTestI18n(language)}>
      <ThemeProvider specifiedTheme={theme} themeAction="/actions/theme">
        <Document>
          <body />
        </Document>
      </ThemeProvider>
    </I18nextProvider>
  )
}

describe("Document", () => {
  test.each(["vi", "en"] as const)("sets lang and dir for %s", (language) => {
    expect(renderDocument(language, null)).toContain(
      `<html lang="${language}" dir="ltr">`
    )
  })

  test("renders the stored theme as the html class on the server", () => {
    expect(renderDocument("vi", Theme.DARK)).toContain(
      '<html lang="vi" dir="ltr" class="dark">'
    )
    expect(renderDocument("vi", Theme.LIGHT)).toContain(
      '<html lang="vi" dir="ltr" class="light">'
    )
  })
})
