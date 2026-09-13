import { renderToString } from "react-dom/server"
import { I18nextProvider } from "react-i18next"
import { describe, expect, test } from "vitest"

import { Document } from "~/components/document"

import { createTestI18n } from "../../tests/utils/i18n"

describe("Document", () => {
  test.each(["vi", "en"] as const)("sets lang and dir for %s", (language) => {
    const html = renderToString(
      <I18nextProvider i18n={createTestI18n(language)}>
        <Document>
          <body />
        </Document>
      </I18nextProvider>
    )

    expect(html).toContain(`<html lang="${language}" dir="ltr">`)
  })
})
