import { describe, expect, test } from "vitest"

import { cn } from "~/lib/utils"

describe("cn", () => {
  test("keeps the last conflicting Tailwind class", () => {
    expect(cn("px-2 py-1", "px-4")).toBe("py-1 px-4")
  })

  test("drops falsy values", () => {
    expect(cn("block", false, undefined, "text-sm")).toBe("block text-sm")
  })
})
