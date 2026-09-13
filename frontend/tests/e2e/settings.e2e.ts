import type { Page, Response } from "@playwright/test"

import { expect, test } from "./support/fixtures"

function actionPosted(page: Page, pathPrefix: string): Promise<Response> {
  return page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      new URL(response.url()).pathname.startsWith(pathPrefix)
  )
}

test("changing the language applies now and after a reload", async ({
  signedInPage: page,
}) => {
  await page.goto("/settings")
  await expect(page.getByRole("heading", { name: "Cài đặt" })).toBeVisible()

  const saved = actionPosted(page, "/actions/locale")
  await page.getByRole("combobox", { name: "Ngôn ngữ" }).click()
  await page.getByRole("option", { name: "English" }).click()
  expect((await saved).ok()).toBe(true)
  await expect(page.getByRole("heading", { name: "Settings" })).toBeVisible()

  await page.reload()
  await expect(page.getByRole("heading", { name: "Settings" })).toBeVisible()
  await expect(page.locator("html")).toHaveAttribute("lang", "en")
})

test("choosing a theme applies now, persists, and can follow the system again", async ({
  signedInPage: page,
}) => {
  await page.emulateMedia({ colorScheme: "light" })
  await page.goto("/settings")
  const html = page.locator("html")
  await expect(html).not.toHaveClass(/\bdark\b/)

  const dark = actionPosted(page, "/actions/theme")
  await page.getByRole("radio", { name: "Tối" }).click()
  expect((await dark).ok()).toBe(true)
  await expect(html).toHaveClass(/\bdark\b/)
  await page.reload()
  await expect(html).toHaveClass(/\bdark\b/)
  await expect(page.getByRole("radio", { name: "Tối" })).toBeChecked()

  const system = actionPosted(page, "/actions/theme")
  await page.getByRole("radio", { name: "Theo hệ thống" }).click()
  expect((await system).ok()).toBe(true)
  await page.reload()
  await expect(html).not.toHaveClass(/\bdark\b/)
  await expect(page.getByRole("radio", { name: "Theo hệ thống" })).toBeChecked()
})
