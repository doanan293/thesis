import { http, HttpResponse } from "msw"
import { createRoutesStub } from "react-router"
import { beforeEach, describe, expect, test } from "vitest"
import { page } from "vitest/browser"
import { render } from "vitest-browser-react"

import type { UserRead } from "~/api/gen/schemas"
import { getListConversationsMockHandler } from "~/api/gen/endpoints.msw"
import { SidebarProvider } from "~/components/ui/sidebar"
import { TooltipProvider } from "~/components/ui/tooltip"
import { AppSidebar } from "~/features/shell/AppSidebar"

import { worker } from "../../../tests/msw/browser"
import { TestProviders } from "../../../tests/utils/providers"

const USER: UserRead = {
  id: "u1",
  email: "an@example.com",
  display_name: "An",
}

function ShellRoute() {
  return (
    <SidebarProvider>
      <AppSidebar user={USER} />
    </SidebarProvider>
  )
}

function LoginRoute() {
  return <p>login-page</p>
}

const Stub = createRoutesStub([
  { path: "/chat", Component: ShellRoute },
  { path: "/login", Component: LoginRoute },
])

async function renderShell() {
  await render(
    <TestProviders>
      <TooltipProvider>
        <Stub initialEntries={["/chat"]} />
      </TooltipProvider>
    </TestProviders>
  )
}

describe("AppSidebar", () => {
  beforeEach(async () => {
    await page.viewport(1280, 800)
    worker.use(
      getListConversationsMockHandler({ items: [], next_cursor: null })
    )
  })

  test("links the private pages and marks the current one", async () => {
    await renderShell()

    // Base UI renders a true boolean state as a bare `data-active` attribute (empty value).
    await expect
      .element(page.getByRole("link", { name: "Trò chuyện" }))
      .toHaveAttribute("data-active")
    await expect
      .element(page.getByRole("link", { name: "Kỹ năng" }))
      .not.toHaveAttribute("data-active")
    await expect
      .element(page.getByRole("link", { name: "Kỹ năng" }))
      .toHaveAttribute("href", "/skills")
    await expect
      .element(page.getByRole("link", { name: "Cài đặt" }))
      .toHaveAttribute("href", "/settings")
    // The shell also mounts the conversation list; wait for its request to finish inside the test,
    // otherwise it reaches MSW after the handlers reset and fails the next test.
    await expect
      .element(page.getByText("Chưa có cuộc trò chuyện nào"))
      .toBeVisible()
  })

  test("logs out through the cookie endpoint and returns to the login page", async () => {
    let loggedOut = false
    worker.use(
      http.post("/api/v1/auth/cookie/logout", () => {
        loggedOut = true
        return new HttpResponse(null, { status: 204 })
      })
    )
    await renderShell()
    await expect
      .element(page.getByText("Chưa có cuộc trò chuyện nào"))
      .toBeVisible()

    await page.getByRole("button", { name: "Menu tài khoản" }).click()
    await page.getByRole("menuitem", { name: "Đăng xuất" }).click()

    await expect.element(page.getByText("login-page")).toBeVisible()
    expect(loggedOut).toBe(true)
  })
})
