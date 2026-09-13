import { useTranslation } from "react-i18next"
import { useNavigate } from "react-router"
import { describe, expect, test } from "vitest"
import { page } from "vitest/browser"

import { toast } from "~/components/ui/toast"

import {
  apiAssistantMessage,
  conversationView,
  messagePage,
  pharmaSource,
} from "../../../tests/chat/fixtures"
import { renderRoutes } from "../../../tests/chat/render"

function Probe() {
  const { t } = useTranslation("chat")
  const navigate = useNavigate()
  return (
    <div>
      <button
        type="button"
        onClick={() => void navigate("/login?next=%2Fchat")}
      >
        {t("composer.send")}
      </button>
      <button
        type="button"
        onClick={() => toast.add({ title: t("notPersisted") })}
      >
        {t("composer.stop")}
      </button>
    </div>
  )
}

describe("chat test support", () => {
  test("renders chat copy in a router with toasts and reports the location", async () => {
    const { queryClient } = await renderRoutes(Probe, {
      initialEntry: "/chat/c1",
    })
    await expect
      .element(page.getByTestId("location"))
      .toHaveTextContent("/chat/c1")
    await page.getByRole("button", { name: "Dừng" }).click()
    await expect
      .element(page.getByText("Không lưu được lượt này"))
      .toBeVisible()
    await page.getByRole("button", { name: "Gửi" }).click()
    await expect
      .element(page.getByTestId("location"))
      .toHaveTextContent("/login?next=%2Fchat")
    expect(queryClient.getDefaultOptions().queries?.retry).toBe(false)
  })

  test("fixtures follow the generated API types", () => {
    const history = messagePage(
      [apiAssistantMessage("a1", "Liều [1].", [pharmaSource(1)])],
      "cursor-1"
    )
    expect(history.next_cursor).toBe("cursor-1")
    expect(history.items[0]?.parts[1]?.type).toBe("source-document")
    expect(conversationView("c1", "Paracetamol").turn_count).toBe(1)
  })
})
