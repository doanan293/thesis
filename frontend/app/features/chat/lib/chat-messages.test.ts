import { describe, expect, test } from "vitest"

import {
  apiAssistantMessage,
  apiUserMessage,
  conversationTurns,
  messagePage,
  pharmaSource,
} from "../../../../tests/chat/fixtures"

import {
  dropTrailingTurn,
  olderMessagesFrom,
  prependOlder,
  questionBefore,
  textOf,
  toPharmaMessage,
  withFeedback,
} from "./chat-messages"

describe("toPharmaMessage", () => {
  test("keeps text, source documents and metadata", () => {
    const message = toPharmaMessage(
      apiAssistantMessage("a1", "Liều 500 mg [1].", [pharmaSource(1)])
    )
    expect(message.role).toBe("assistant")
    expect(message.parts[0]).toEqual({ type: "text", text: "Liều 500 mg [1]." })
    expect(message.parts[1]).toMatchObject({
      type: "source-document",
      sourceId: "chunk-1",
      providerMetadata: { pharma: { index: 1, isCurrent: true } },
    })
    expect(message.metadata?.status).toBe("completed")
  })
})

describe("history merge", () => {
  test("prepends older messages and drops ids already present", () => {
    const current = conversationTurns(2, "new").map(toPharmaMessage)
    const older = [
      ...conversationTurns(1, "old"),
      apiUserMessage("new-u1", "trùng"),
    ].map(toPharmaMessage)
    expect(prependOlder(current, older).map((message) => message.id)).toEqual([
      "old-u1",
      "old-a1",
      "new-u1",
      "new-a1",
      "new-u2",
      "new-a2",
    ])
  })

  test("returns pages not applied yet, oldest first", () => {
    const pages = [
      messagePage(conversationTurns(1, "p0"), "cursor-1"),
      messagePage(conversationTurns(1, "p1"), "cursor-2"),
      messagePage(conversationTurns(1, "p2")),
    ]
    expect(olderMessagesFrom(pages, 1).map((message) => message.id)).toEqual([
      "p2-u1",
      "p2-a1",
      "p1-u1",
      "p1-a1",
    ])
    expect(olderMessagesFrom(pages, 3)).toEqual([])
  })
})

describe("message helpers", () => {
  const messages = [
    apiUserMessage("u1", "Liều paracetamol?"),
    apiAssistantMessage("a1", "Người lớn 500 mg [1].", [pharmaSource(1)]),
  ].map(toPharmaMessage)

  test("joins text parts", () => {
    const [, assistant] = messages
    expect(assistant === undefined ? "" : textOf(assistant)).toBe(
      "Người lớn 500 mg [1]."
    )
  })

  test("finds the question before an assistant message", () => {
    expect(questionBefore(messages, "a1")).toBe("Liều paracetamol?")
    expect(questionBefore(messages, "missing")).toBeUndefined()
  })

  test("sets feedback on one message", () => {
    const updated = withFeedback(messages, "a1", {
      rating: "down",
      note: "Thiếu liều trẻ em",
    })
    expect(updated[1]?.metadata?.feedback).toEqual({
      rating: "down",
      note: "Thiếu liều trẻ em",
    })
    expect(updated[0]).toBe(messages[0])
  })

  test("drops the trailing user and assistant pair", () => {
    expect(dropTrailingTurn(messages)).toEqual([])
    expect(dropTrailingTurn(messages.slice(0, 1))).toEqual([])
    expect(dropTrailingTurn([])).toEqual([])
  })
})
