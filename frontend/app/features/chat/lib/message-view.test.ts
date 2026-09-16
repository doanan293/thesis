import { describe, expect, test } from "vitest"

import {
  apiAssistantMessage,
  apiUserMessage,
  pharmaSource,
} from "../../../../tests/chat/fixtures"

import { toPharmaMessage } from "./chat-messages"
import type { PharmaUIMessage } from "./message-schema"
import { describeMessage } from "./message-view"

const IDLE = { streaming: false, failed: false }

function streamingAssistant(parts: PharmaUIMessage["parts"]): PharmaUIMessage {
  return { id: "a1", role: "assistant", parts }
}

describe("describeMessage", () => {
  test("maps a streaming message before text arrives", () => {
    const view = describeMessage(
      streamingAssistant([
        { type: "data-phase", id: "phase", data: { phase: "understanding" } },
        {
          type: "data-phase",
          id: "phase",
          data: { phase: "searching", round: 2 },
        },
        {
          type: "data-evidence",
          id: "evidence",
          data: {
            items: [
              {
                index: 1,
                source: "Dược thư",
                title: "Paracetamol",
                section: "Liều dùng",
                startPage: 812,
                endPage: 813,
                snippet: "Người lớn…",
              },
            ],
          },
        },
      ]),
      { streaming: true, failed: false }
    )
    expect(view.showStatus).toBe(true)
    expect(view.phase).toEqual({ phase: "searching", round: 2 })
    expect(view.evidence).toHaveLength(1)
    expect(view.hasText).toBe(false)
    expect(view.notice).toBeNull()
    expect(view.canGiveFeedback).toBe(false)
  })

  test("maps a completed history message with sorted sources", () => {
    const message = toPharmaMessage(
      apiAssistantMessage(
        "a1",
        "Liều [2] [1].",
        [pharmaSource(2), pharmaSource(1)],
        {
          feedback: { rating: "up", note: "" },
        }
      )
    )
    const view = describeMessage(message, IDLE)
    expect(view.text).toBe("Liều [2] [1].")
    expect(view.sources.map((source) => source.index)).toEqual([1, 2])
    expect(view.showStatus).toBe(false)
    expect(view.feedback).toEqual({ rating: "up", note: "" })
    expect(view.canGiveFeedback).toBe(true)
  })

  test("maps guardrail and error statuses and keeps abstained as an answer", () => {
    const of = (
      status: "blocked" | "redirected" | "timeout" | "error" | "abstained"
    ) =>
      describeMessage(
        toPharmaMessage(
          apiAssistantMessage(`a-${status}`, "Nội dung.", [], { status })
        ),
        IDLE
      )
    expect(of("blocked").notice).toEqual({
      kind: "guardrail",
      reason: "blocked",
    })
    expect(of("redirected").notice).toEqual({
      kind: "guardrail",
      reason: "redirected",
    })
    expect(of("timeout").notice).toEqual({ kind: "error", reason: "timeout" })
    expect(of("error").notice).toEqual({ kind: "error", reason: "failed" })
    expect(of("timeout").canGiveFeedback).toBe(false)
    expect(of("abstained").notice).toBeNull()
  })

  test("maps a transport failure without metadata to a network error", () => {
    const view = describeMessage(
      streamingAssistant([{ type: "text", text: "Đang" }]),
      {
        streaming: false,
        failed: true,
      }
    )
    expect(view.notice).toEqual({ kind: "error", reason: "network" })
  })

  test("refuses feedback on a turn that was not persisted", () => {
    const message = toPharmaMessage(
      apiAssistantMessage("a1", "Liều.", [], { persisted: false })
    )
    expect(describeMessage(message, IDLE).canGiveFeedback).toBe(false)
  })

  test("maps a user message to its text", () => {
    const view = describeMessage(
      toPharmaMessage(apiUserMessage("u1", "Liều paracetamol?")),
      IDLE
    )
    expect(view.text).toBe("Liều paracetamol?")
    expect(view.canGiveFeedback).toBe(false)
  })
})
