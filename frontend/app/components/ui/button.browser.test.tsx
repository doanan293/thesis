import { expect, test, vi } from "vitest"
import { render } from "vitest-browser-react"

import { Button } from "~/components/ui/button"

test("Button renders a native button and reports clicks", async () => {
  const onClick = vi.fn<() => void>()
  const screen = await render(<Button onClick={onClick}>Gửi</Button>)

  await screen.getByRole("button", { name: "Gửi" }).click()

  expect(onClick).toHaveBeenCalledOnce()
})
