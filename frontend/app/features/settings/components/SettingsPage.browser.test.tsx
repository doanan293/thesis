import { http, HttpResponse } from "msw"
import { createRoutesStub } from "react-router"
import { ThemeProvider } from "remix-themes"
import { describe, expect, test } from "vitest"
import { page } from "vitest/browser"
import { render } from "vitest-browser-react"
import { z } from "zod"

import { getUsersCurrentUserMockHandler } from "~/api/gen/endpoints.msw"
import type { UserRead } from "~/api/gen/schemas"
import { Toaster } from "~/components/ui/toast"

import { worker } from "../../../../tests/msw/browser"
import { TestProviders } from "../../../../tests/utils/providers"
import { SettingsPage } from "./SettingsPage"

const USER: UserRead = {
  id: "0b7c3f5e-4a53-4f5e-9a51-7d1e2f3a4b5c",
  email: "an@example.com",
  is_active: true,
  is_superuser: false,
  is_verified: false,
  display_name: "Nguyễn An",
}

const ThemeActionBody = z.object({
  theme: z.enum(["light", "dark"]).nullable(),
})
type ThemeActionBody = z.infer<typeof ThemeActionBody>

async function renderSettings(): Promise<{
  locales: string[]
  themes: ThemeActionBody[]
}> {
  const locales: string[] = []
  const themes: ThemeActionBody[] = []
  worker.use(
    getUsersCurrentUserMockHandler(USER),
    http.post("*/actions/theme", async ({ request }) => {
      themes.push(ThemeActionBody.parse(await request.json()))
      return HttpResponse.json({ success: true })
    })
  )
  const Stub = createRoutesStub([
    { path: "/settings", Component: SettingsPage },
    {
      path: "/actions/locale",
      action: async ({ request }) => {
        const locale = (await request.formData()).get("locale")
        locales.push(typeof locale === "string" ? locale : "")
        return { ok: true }
      },
    },
  ])
  await render(
    <TestProviders>
      <ThemeProvider specifiedTheme={null} themeAction="/actions/theme">
        <Toaster>
          <Stub initialEntries={["/settings"]} />
        </Toaster>
      </ThemeProvider>
    </TestProviders>
  )
  return { locales, themes }
}

describe("SettingsPage", () => {
  test("renders profile, password, language and theme sections", async () => {
    await renderSettings()

    await expect
      .element(page.getByRole("heading", { name: "Cài đặt" }))
      .toBeVisible()
    await expect
      .element(page.getByLabelText("Tên hiển thị"))
      .toHaveValue("Nguyễn An")
    await expect
      .element(page.getByLabelText("Mật khẩu mới", { exact: true }))
      .toBeVisible()
    await expect
      .element(page.getByRole("combobox", { name: "Ngôn ngữ" }))
      .toMatchTextContent("Tiếng Việt")
    await expect
      .element(page.getByRole("radio", { name: "Theo hệ thống" }))
      .toBeChecked()
  })

  test("switches the language and posts it to the locale action", async () => {
    const { locales } = await renderSettings()

    await page.getByRole("combobox", { name: "Ngôn ngữ" }).click()
    await page.getByRole("option", { name: "English" }).click()

    await expect
      .element(page.getByRole("heading", { name: "Settings" }))
      .toBeVisible()
    await expect.poll(() => locales).toEqual(["en"])
  })

  test("stores the chosen theme and can return to the system theme", async () => {
    const { themes } = await renderSettings()

    await page.getByRole("radio", { name: "Tối" }).click()
    await expect.element(page.getByRole("radio", { name: "Tối" })).toBeChecked()
    await expect.poll(() => themes).toEqual([{ theme: "dark" }])

    await page.getByRole("radio", { name: "Theo hệ thống" }).click()
    await expect
      .element(page.getByRole("radio", { name: "Theo hệ thống" }))
      .toBeChecked()
    await expect
      .poll(() => themes)
      .toEqual([{ theme: "dark" }, { theme: null }])
  })
})
