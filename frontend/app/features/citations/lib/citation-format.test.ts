import { describe, expect, test } from "vitest"

import { pageLabel } from "./citation-format"

describe("pageLabel", () => {
  test("formats single pages and ranges", () => {
    expect(pageLabel(812, 813)).toEqual({
      key: "pages.range",
      start: 812,
      end: 813,
    })
    expect(pageLabel(812, 812)).toEqual({ key: "pages.single", page: 812 })
    expect(pageLabel(812, null)).toEqual({ key: "pages.single", page: 812 })
    expect(pageLabel(null, 813)).toEqual({ key: "pages.single", page: 813 })
    expect(pageLabel(null, null)).toBeNull()
  })
})
