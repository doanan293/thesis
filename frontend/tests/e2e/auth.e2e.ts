import type { Page } from "@playwright/test"

import { type TestUser, newUser } from "./support/api"
import { expect, test } from "./support/fixtures"

async function logIn(page: Page, user: TestUser): Promise<void> {
  await page.getByLabel("Email", { exact: true }).fill(user.email)
  await page.getByLabel("Mật khẩu", { exact: true }).fill(user.password)
  await page.getByRole("button", { name: "Đăng nhập" }).click()
}

async function logOut(page: Page): Promise<void> {
  await page.getByRole("button", { name: "Menu tài khoản" }).click()
  await page.getByRole("menuitem", { name: "Đăng xuất" }).click()
  await expect(page).toHaveURL(/\/login(\?|$)/)
}

test("register, log out, log in and log out again", async ({ page }) => {
  const user = newUser()

  await page.goto("/register")
  await page.getByLabel("Tên hiển thị", { exact: true }).fill(user.displayName)
  await page.getByLabel("Email", { exact: true }).fill(user.email)
  await page.getByLabel("Mật khẩu", { exact: true }).fill(user.password)
  await page.getByRole("button", { name: "Tạo tài khoản" }).click()
  // P8's register form signs the new user in and opens /chat.
  await expect(page).toHaveURL(/\/chat$/)
  await logOut(page)

  await logIn(page, user)
  await expect(page).toHaveURL(/\/chat$/)
  await logOut(page)

  await page.goto("/settings")
  await expect(page).toHaveURL(/\/login\?next=/)
})

test("an expired session returns to login and then back to the page", async ({
  signedInPage: page,
  user,
}) => {
  await page.goto("/chat")
  await expect(
    page.getByRole("textbox", { name: "Tin nhắn", exact: true })
  ).toBeVisible()

  // The browser no longer has a valid session, exactly as after the lifetime ends.
  await page.context().clearCookies({ name: "pharma_session" })
  await page.getByRole("link", { name: "Kỹ năng", exact: true }).click()

  await expect(page).toHaveURL(/\/login\?next=(%2F|\/)skills/)
  await logIn(page, user)
  await expect(page).toHaveURL(/\/skills$/)
})
