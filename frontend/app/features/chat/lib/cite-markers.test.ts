import { describe, expect, test } from "vitest"

import { parseCiteIndex, toCiteRefMarkup } from "./cite-markers"

describe("toCiteRefMarkup", () => {
  test("converts every complete marker", () => {
    expect(toCiteRefMarkup("Người lớn 500 mg [1] và trẻ em [12].")).toBe(
      'Người lớn 500 mg <cite-ref index="1"></cite-ref> và trẻ em <cite-ref index="12"></cite-ref>.'
    )
  })

  test("converts adjacent markers", () => {
    expect(toCiteRefMarkup("[1][2]")).toBe(
      '<cite-ref index="1"></cite-ref><cite-ref index="2"></cite-ref>'
    )
  })

  test("leaves links, words and partial markers alone", () => {
    expect(toCiteRefMarkup("[1](https://example.com)")).toBe(
      "[1](https://example.com)"
    )
    expect(toCiteRefMarkup("[a] [1")).toBe("[a] [1")
    expect(toCiteRefMarkup("[1234]")).toBe("[1234]")
  })
})

describe("parseCiteIndex", () => {
  test("accepts a positive integer string only", () => {
    expect(parseCiteIndex("3")).toBe(3)
    expect(parseCiteIndex("0")).toBeNull()
    expect(parseCiteIndex("x")).toBeNull()
    expect(parseCiteIndex(3)).toBeNull()
    expect(parseCiteIndex(undefined)).toBeNull()
  })
})
