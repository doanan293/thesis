import { http, HttpResponse } from "msw"
import { createRoutesStub } from "react-router"
import { beforeEach, describe, expect, test } from "vitest"
import { page } from "vitest/browser"
import { render } from "vitest-browser-react"

import { RegisterForm } from "~/features/auth/RegisterForm"
import { applyZodLocale } from "~/i18n/zod-locale"

import { worker } from "../../../tests/msw/browser"
import { TestProviders } from "../../../tests/utils/providers"

function RegisterRoute() {
  return <RegisterForm nextPath="/chat" />
}

function ChatRoute() {
  return <p>chat-page</p>
}

const Stub = createRoutesStub([
  { path: "/register", Component: RegisterRoute },
  { path: "/chat", Component: ChatRoute },
])

async function renderAndSubmit(password = "correct horse") {
  await render(
    <TestProviders>
      <Stub initialEntries={["/register"]} />
    </TestProviders>
  )
  await page.getByLabelText("Tên hiển thị").fill("An")
  await page.getByLabelText("Email").fill("an@example.com")
  await page.getByLabelText("Mật khẩu").fill(password)
  await page.getByRole("button", { name: "Tạo tài khoản" }).click()
}

describe("RegisterForm", () => {
  beforeEach(() => {
    applyZodLocale("vi")
  })

  test("registers, signs in with the same credentials and opens the next path", async () => {
    let registered: unknown = null
    let loginBody = ""
    worker.use(
      http.post("/api/v1/auth/register", async ({ request }) => {
        registered = await request.json()
        return HttpResponse.json(
          { id: "u1", email: "an@example.com", display_name: "An" },
          { status: 201 }
        )
      }),
      http.post("/api/v1/auth/cookie/login", async ({ request }) => {
        loginBody = await request.text()
        return new HttpResponse(null, { status: 204 })
      }),
      http.get("/api/v1/users/me", () =>
        HttpResponse.json({ id: "u1", email: "an@example.com" })
      )
    )

    await renderAndSubmit()

    await expect.element(page.getByText("chat-page")).toBeVisible()
    expect(registered).toEqual({
      email: "an@example.com",
      password: "correct horse",
      display_name: "An",
    })
    expect(new URLSearchParams(loginBody).get("username")).toBe(
      "an@example.com"
    )
  })

  test("requires a password of at least 8 characters before calling the API", async () => {
    // No handler is registered: MSW's onUnhandledRequest "error" fails the test if a request is sent.
    await renderAndSubmit("1234567")

    await expect
      .element(page.getByText("Mật khẩu cần ít nhất 8 ký tự."))
      .toBeVisible()
  })

  test("shows the translated message when the email is taken", async () => {
    worker.use(
      http.post("/api/v1/auth/register", () =>
        HttpResponse.json(
          {
            type: "urn:pharma-agent:problem:register-user-already-exists",
            title: "User already exists",
            status: 400,
            code: "REGISTER_USER_ALREADY_EXISTS",
          },
          {
            status: 400,
            headers: { "Content-Type": "application/problem+json" },
          }
        )
      )
    )

    await renderAndSubmit()

    await expect
      .element(page.getByRole("alert"))
      .toHaveTextContent("Email này đã được đăng ký.")
  })
})
