import { renderToString } from "react-dom/server"
import { I18nextProvider } from "react-i18next"
import { MemoryRouter } from "react-router"
import { describe, expect, test } from "vitest"

import NotFound, { loader } from "~/routes/not-found"

import { createTestI18n } from "../../tests/utils/i18n"

describe("not-found route", () => {
  test("answers with HTTP 404", () => {
    expect(loader().init?.status).toBe(404)
  })

  test("renders a translated message with a link home", () => {
    const html = renderToString(
      <I18nextProvider i18n={createTestI18n("vi")}>
        <MemoryRouter>
          <NotFound />
        </MemoryRouter>
      </I18nextProvider>
    )

    expect(html).toContain("Không tìm thấy trang")
    expect(html).toContain('href="/"')
  })
})
