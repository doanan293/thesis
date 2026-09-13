import { http, HttpResponse } from "msw"
import { describe, expect, test } from "vitest"
import { page } from "vitest/browser"
import { render } from "vitest-browser-react"

import { getUsersPatchCurrentUserMockHandler } from "~/api/gen/endpoints.msw"
import type { UserRead, UserUpdate } from "~/api/gen/schemas"
import { UsersPatchCurrentUserBody } from "~/api/gen/zod"
import { Toaster } from "~/components/ui/toast"

import { worker } from "../../../../tests/msw/browser"
import { TestProviders } from "../../../../tests/utils/providers"
import { ProfileForm } from "./ProfileForm"

const USER: UserRead = {
  id: "0b7c3f5e-4a53-4f5e-9a51-7d1e2f3a4b5c",
  email: "an@example.com",
  is_active: true,
  is_superuser: false,
  is_verified: false,
  display_name: "Nguyễn An",
}

async function renderForm(): Promise<void> {
  await render(
    <TestProviders>
      <Toaster>
        <ProfileForm user={USER} />
      </Toaster>
    </TestProviders>
  )
}

describe("ProfileForm", () => {
  test("saves the trimmed display name", async () => {
    const bodies: UserUpdate[] = []
    worker.use(
      getUsersPatchCurrentUserMockHandler(async ({ request }) => {
        const body = UsersPatchCurrentUserBody.parse(await request.json())
        bodies.push(body)
        return { ...USER, display_name: body.display_name ?? "" }
      })
    )
    await renderForm()
    const input = page.getByLabelText("Tên hiển thị")
    await expect.element(input).toHaveValue("Nguyễn An")

    await input.fill("  Bác sĩ An  ")
    await page.getByRole("button", { name: "Lưu" }).click()

    await expect.element(page.getByText("Đã lưu tên hiển thị.")).toBeVisible()
    await expect.element(input).toHaveValue("Bác sĩ An")
    expect(bodies).toEqual([{ display_name: "Bác sĩ An" }])
  })

  test("requires a non-blank name without calling the API", async () => {
    await renderForm()

    await page.getByLabelText("Tên hiển thị").fill("   ")
    await page.getByRole("button", { name: "Lưu" }).click()

    await expect
      .element(page.getByRole("alert"))
      .toHaveTextContent("Hãy nhập tên hiển thị.")
  })

  test("puts a 422 item for display_name on the field", async () => {
    worker.use(
      http.patch("*/api/v1/users/me", () =>
        HttpResponse.json(
          {
            type: "urn:pharma-agent:problem:validation-error",
            title: "Validation error",
            status: 422,
            code: "VALIDATION_ERROR",
            errors: [
              {
                loc: ["body", "display_name"],
                message: "Value is not allowed",
                type: "value_error",
              },
            ],
          },
          {
            status: 422,
            headers: { "Content-Type": "application/problem+json" },
          }
        )
      )
    )
    await renderForm()

    await page.getByLabelText("Tên hiển thị").fill("Tên mới")
    await page.getByRole("button", { name: "Lưu" }).click()

    await expect
      .element(page.getByRole("alert"))
      .toHaveTextContent("Value is not allowed")
  })
})
