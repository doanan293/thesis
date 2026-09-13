import { http, HttpResponse } from "msw"
import { describe, expect, test } from "vitest"
import { page } from "vitest/browser"
import { render } from "vitest-browser-react"

import {
  getUsersPatchCurrentUserMockHandler,
  getUsersPatchCurrentUserResponseMock,
} from "~/api/gen/endpoints.msw"
import type { UserUpdate } from "~/api/gen/schemas"
import { UsersPatchCurrentUserBody } from "~/api/gen/zod"
import { Toaster } from "~/components/ui/toast"

import { worker } from "../../../../tests/msw/browser"
import { TestProviders } from "../../../../tests/utils/providers"
import { PasswordForm } from "./PasswordForm"

async function renderForm(): Promise<void> {
  await render(
    <TestProviders>
      <Toaster>
        <PasswordForm />
      </Toaster>
    </TestProviders>
  )
}

async function fillPasswords(value: string, confirm: string): Promise<void> {
  await page.getByLabelText("Mật khẩu mới", { exact: true }).fill(value)
  await page
    .getByLabelText("Nhập lại mật khẩu mới", { exact: true })
    .fill(confirm)
  await page.getByRole("button", { name: "Đổi mật khẩu" }).click()
}

describe("PasswordForm", () => {
  test("changes the password and clears the fields", async () => {
    const bodies: UserUpdate[] = []
    worker.use(
      getUsersPatchCurrentUserMockHandler(async ({ request }) => {
        bodies.push(UsersPatchCurrentUserBody.parse(await request.json()))
        return getUsersPatchCurrentUserResponseMock()
      })
    )
    await renderForm()

    await fillPasswords("mat-khau-moi-123", "mat-khau-moi-123")

    await expect.element(page.getByText("Đã đổi mật khẩu.")).toBeVisible()
    await expect
      .element(page.getByLabelText("Mật khẩu mới", { exact: true }))
      .toHaveValue("")
    expect(bodies).toEqual([{ password: "mat-khau-moi-123" }])
  })

  test("requires at least 8 characters without calling the API", async () => {
    await renderForm()

    await fillPasswords("1234567", "1234567")

    await expect
      .element(page.getByRole("alert"))
      .toHaveTextContent("Mật khẩu tối thiểu 8 ký tự.")
  })

  test("reports a mismatch without calling the API", async () => {
    await renderForm()

    await fillPasswords("mat-khau-moi-123", "mat-khau-khac-123")

    await expect
      .element(page.getByRole("alert"))
      .toHaveTextContent("Hai mật khẩu không khớp.")
  })

  test("puts the server password policy error on the password field", async () => {
    worker.use(
      http.patch("*/api/v1/users/me", () =>
        HttpResponse.json(
          {
            type: "urn:pharma-agent:problem:update-user-invalid-password",
            title: "Bad request",
            status: 400,
            code: "UPDATE_USER_INVALID_PASSWORD",
            detail: "Password should not contain e-mail",
          },
          {
            status: 400,
            headers: { "Content-Type": "application/problem+json" },
          }
        )
      )
    )
    await renderForm()

    await fillPasswords("an@example.com1", "an@example.com1")

    await expect
      .element(page.getByRole("alert"))
      .toHaveTextContent("Mật khẩu mới không đạt yêu cầu.")
  })
})
