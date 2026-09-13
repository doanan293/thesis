import { describe, expect, test } from "vitest"

import {
  apiAssistantMessage,
  pharmaSource,
} from "../../../../tests/chat/fixtures"

import { toPharmaMessage } from "./chat-messages"
import { citationSourcesOf, readPharmaSource } from "./sources"

describe("sources", () => {
  test("indexes source documents by citation index", () => {
    const message = toPharmaMessage(
      apiAssistantMessage("a1", "[1] [2]", [
        pharmaSource(1),
        pharmaSource(2, { isCurrent: false }),
      ])
    )
    const sources = citationSourcesOf(message)
    expect([...sources.keys()]).toEqual([1, 2])
    expect(sources.get(2)?.isCurrent).toBe(false)
  })

  test("ignores a source document without pharma metadata", () => {
    expect(
      readPharmaSource({
        type: "source-document",
        sourceId: "x",
        mediaType: "text/markdown",
        title: "x",
      })
    ).toBeUndefined()
  })
})
