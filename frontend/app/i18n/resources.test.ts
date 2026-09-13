import { describe, expect, test } from "vitest"

import { NAMESPACES, resources } from "~/i18n/config"

function keysOf(value: unknown, prefix = ""): string[] {
  if (typeof value !== "object" || value === null) {
    return [prefix]
  }
  return Object.entries(value).flatMap(([key, child]) =>
    keysOf(child, prefix === "" ? key : `${prefix}.${key}`)
  )
}

describe("translation resources", () => {
  test("both languages register every namespace", () => {
    const expected = [...NAMESPACES].toSorted()
    expect(Object.keys(resources.vi).toSorted()).toEqual(expected)
    expect(Object.keys(resources.en).toSorted()).toEqual(expected)
  })

  test.each(NAMESPACES)("vi and en define the same keys in %s", (namespace) => {
    expect(keysOf(resources.en[namespace]).toSorted()).toEqual(
      keysOf(resources.vi[namespace]).toSorted()
    )
  })
})
