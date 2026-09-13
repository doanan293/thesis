import type { QueryClient } from "@tanstack/react-query"
import { delay, http, HttpResponse } from "msw"
import { describe, expect, test } from "vitest"
import { page } from "vitest/browser"
import { render } from "vitest-browser-react"

import {
  getDeleteSkillMockHandler,
  getListSkillsMockHandler,
  getSetSkillEnabledMockHandler,
} from "~/api/gen/endpoints.msw"
import type { SkillView } from "~/api/gen/schemas"
import { SetSkillEnabledBody } from "~/api/gen/zod"
import { Toaster } from "~/components/ui/toast"

import { worker } from "../../../../tests/msw/browser"
import {
  createTestQueryClient,
  TestProviders,
} from "../../../../tests/utils/providers"
import { SkillsPage } from "./SkillsPage"

function skill(overrides: Partial<SkillView>): SkillView {
  return {
    name: "drug-monograph",
    title: "Chuyên luận thuốc",
    description: "Tra cứu theo chuyên luận Dược thư.",
    enabled: true,
    is_system: true,
    version: "0123456789ab",
    ...overrides,
  }
}

const SYSTEM = skill({})
const OWN = skill({
  name: "my-notes",
  title: "Ghi chú của tôi",
  description: "Ưu tiên liều cho trẻ em.",
  is_system: false,
  version: "ba9876543210",
})

/** Serves GET/PATCH/DELETE /skills from an in-memory list so refetches see changes. */
function serveSkills(
  initial: SkillView[],
  patchDelayMs = 0
): { current: () => SkillView[] } {
  let stored = [...initial]
  worker.use(
    getListSkillsMockHandler(() => stored),
    getSetSkillEnabledMockHandler(async ({ params, request }) => {
      await delay(patchDelayMs)
      const body = SetSkillEnabledBody.parse(await request.json())
      const name = String(params.name)
      stored = stored.map((item) =>
        item.name === name ? { ...item, enabled: body.enabled } : item
      )
      return stored.find((item) => item.name === name) ?? skill({ name })
    }),
    getDeleteSkillMockHandler(({ params }) => {
      stored = stored.filter((item) => item.name !== String(params.name))
    })
  )
  return { current: () => stored }
}

async function renderPage(): Promise<QueryClient> {
  const queryClient = createTestQueryClient()
  await render(
    <TestProviders client={queryClient}>
      <Toaster>
        <SkillsPage />
      </Toaster>
    </TestProviders>
  )
  return queryClient
}

/**
 * The toggle and delete mutations refetch the list when they settle. Waiting
 * for that inside the test keeps the refetch from reaching MSW after its
 * handlers are reset for the next test.
 */
async function settled(queryClient: QueryClient): Promise<void> {
  await expect
    .poll(() => queryClient.isFetching() + queryClient.isMutating())
    .toBe(0)
}

describe("SkillsPage", () => {
  test("shows system skills read-only and own skills with controls", async () => {
    serveSkills([SYSTEM, OWN])
    await renderPage()

    await expect
      .element(page.getByRole("heading", { name: "Kỹ năng hệ thống" }))
      .toBeVisible()
    await expect
      .element(page.getByRole("heading", { name: "Kỹ năng của bạn" }))
      .toBeVisible()
    await expect.element(page.getByText("Chuyên luận thuốc")).toBeVisible()
    await expect
      .element(page.getByText("Hệ thống", { exact: true }))
      .toBeVisible()
    await expect
      .element(page.getByRole("switch", { name: "Bật my-notes" }))
      .toBeChecked()
    expect(
      page.getByRole("switch", { name: "Bật drug-monograph" }).query()
    ).toBeNull()
    expect(
      page.getByRole("button", { name: "Xoá drug-monograph" }).query()
    ).toBeNull()
  })

  test("shows the empty state when the user has no skills", async () => {
    serveSkills([SYSTEM])
    await renderPage()

    await expect
      .element(page.getByText("Bạn chưa tải lên kỹ năng nào."))
      .toBeVisible()
  })

  test("flips the switch before the server answers and keeps the saved state", async () => {
    const store = serveSkills([SYSTEM, OWN], 300)
    const queryClient = await renderPage()
    const toggle = page.getByRole("switch", { name: "Bật my-notes" })
    await expect.element(toggle).toBeChecked()

    await toggle.click()

    await expect.element(toggle).not.toBeChecked()
    await expect
      .poll(
        () => store.current().find((item) => item.name === "my-notes")?.enabled
      )
      .toBe(false)
    await settled(queryClient)
    await expect.element(toggle).not.toBeChecked()
  })

  test("rolls the switch back and shows a toast when the server rejects", async () => {
    serveSkills([SYSTEM, OWN])
    worker.use(
      http.patch("*/api/v1/skills/:name", () =>
        HttpResponse.json(
          {
            type: "urn:pharma-agent:problem:skill-not-found",
            title: "Skill not found",
            status: 404,
            code: "SKILL_NOT_FOUND",
            detail: "my-notes",
          },
          {
            status: 404,
            headers: { "Content-Type": "application/problem+json" },
          }
        )
      )
    )
    const queryClient = await renderPage()
    const toggle = page.getByRole("switch", { name: "Bật my-notes" })
    await expect.element(toggle).toBeChecked()

    await toggle.click()

    await expect
      .element(page.getByText("Không đổi được trạng thái kỹ năng my-notes."))
      .toBeVisible()
    await settled(queryClient)
    await expect.element(toggle).toBeChecked()
  })

  test("deletes an own skill after confirmation", async () => {
    const store = serveSkills([SYSTEM, OWN])
    const queryClient = await renderPage()

    await page.getByRole("button", { name: "Xoá my-notes" }).click()
    const dialog = page.getByRole("alertdialog")
    await expect
      .element(dialog.getByText("Xoá kỹ năng my-notes?"))
      .toBeVisible()
    await dialog.getByRole("button", { name: "Xoá", exact: true }).click()

    await expect
      .element(page.getByText("Đã xoá kỹ năng my-notes."))
      .toBeVisible()
    await expect
      .element(page.getByText("Bạn chưa tải lên kỹ năng nào."))
      .toBeVisible()
    await settled(queryClient)
    expect(store.current().map((item) => item.name)).toEqual(["drug-monograph"])
  })

  test("keeps the skill when the dialog is cancelled", async () => {
    const store = serveSkills([SYSTEM, OWN])
    await renderPage()

    await page.getByRole("button", { name: "Xoá my-notes" }).click()
    await page
      .getByRole("alertdialog")
      .getByRole("button", { name: "Huỷ" })
      .click()

    await expect.element(page.getByRole("alertdialog")).not.toBeInTheDocument()
    await expect.element(page.getByText("Ghi chú của tôi")).toBeVisible()
    expect(store.current()).toHaveLength(2)
  })
})
