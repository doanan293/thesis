import { expect, test } from "vitest"
import { page } from "vitest/browser"
import { render } from "vitest-browser-react"

import { Toaster, useToastManager } from "~/components/ui/toast"

function SaveButton() {
  const manager = useToastManager()
  return (
    <button
      type="button"
      onClick={() => {
        manager.add({ title: "Không lưu được lượt này", type: "error" })
      }}
    >
      Lưu
    </button>
  )
}

test("Toaster shows a toast added through its manager", async () => {
  await render(
    <Toaster>
      <SaveButton />
    </Toaster>
  )

  await page.getByRole("button", { name: "Lưu" }).click()

  await expect.element(page.getByText("Không lưu được lượt này")).toBeVisible()
})
