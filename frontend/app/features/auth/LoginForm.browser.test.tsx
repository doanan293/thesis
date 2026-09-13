import { http, HttpResponse } from "msw"
import { createRoutesStub } from "react-router"
import { beforeEach, describe, expect, test } from "vitest"
import { page } from "vitest/browser"
import { render } from "vitest-browser-react"

import { LoginForm } from "~/features/auth/LoginForm"
import { applyZodLocale } from "~/i18n/zod-locale"

import { worker } from "../../../tests/msw/browser"
import { TestProviders } from "../../../tests/utils/providers"

function LoginRoute() {
  return <LoginForm nextPath="/chat" />
}

function ChatRoute() {
  return <p>chat-page</p>
}

const Stub = createRoutesStub([
  { path: "/login", Component: LoginRoute },
  { path: "/chat", Component: ChatRoute },
])

async function renderLogin() {
  await render(
    <TestProviders>
      <Stub initialEntries={["/login"]} />
    </TestProviders>
  )
}

async function submit(email: string, password: string) {
  await page.getByLabelText("Email").fill(email)
  await page.getByLabelText("Mật khẩu").fill(password)
  await page.getByRole("button", { name: "Đăng nhập" }).click()
}

describe("LoginForm", () => {
  beforeEach(() => {
    applyZodLocale("vi")
  })

  test("signs in through the cookie endpoint and opens the next path", async () => {
    let body = ""
    worker.use(
      http.post("/api/v1/auth/cookie/login", async ({ request }) => {
        body = await request.text()
        return new HttpResponse(null, { status: 204 })
      }),
      http.get("/api/v1/users/me", () =>
        HttpResponse.json({ id: "u1", email: "an@example.com" })
      )
    )
    await renderLogin()

    await submit("an@example.com", "correct horse")

    await expect.element(page.getByText("chat-page")).toBeVisible()
    const form = new URLSearchParams(body)
    expect(form.get("username")).toBe("an@example.com")
    expect(form.get("password")).toBe("correct horse")
  })

  test("shows the translated message for bad credentials", async () => {
    worker.use(
      http.post("/api/v1/auth/cookie/login", () =>
        HttpResponse.json(
          {
            type: "urn:pharma-agent:problem:login-bad-credentials",
            title: "Bad credentials",
            status: 400,
            code: "LOGIN_BAD_CREDENTIALS",
          },
          {
            status: 400,
            headers: { "Content-Type": "application/problem+json" },
          }
        )
      )
    )
    await renderLogin()

    await submit("an@example.com", "wrong")

    await expect
      .element(page.getByRole("alert"))
      .toHaveTextContent("Email hoặc mật khẩu không đúng.")
  })

  test("validates the email before calling the API", async () => {
    await renderLogin()

    await submit("not-an-email", "x")

    await expect
      .element(page.getByText("địa chỉ email không hợp lệ"))
      .toBeVisible()
  })
})
