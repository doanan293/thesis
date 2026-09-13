import { createUIMessageStreamResponse, type UIMessageChunk } from "ai"
import { http, HttpResponse } from "msw"
import { afterEach, describe, expect, test } from "vitest"

import {
  apiAssistantMessage,
  apiUserMessage,
} from "../../../../tests/chat/fixtures"
import { worker } from "../../../../tests/msw/browser"

import { toPharmaMessage } from "./chat-messages"
import { CHAT_STREAM_API, createChatTransport } from "./chat-transport"
import { readStreamProblem } from "./stream-problem"

async function readAll<T>(stream: ReadableStream<T>): Promise<T[]> {
  const reader = stream.getReader()
  const values: T[] = []
  let result = await reader.read()
  while (!result.done) {
    values.push(result.value)
    result = await reader.read()
  }
  return values
}

const history = [
  apiUserMessage("u1", "Câu hỏi đầu"),
  apiAssistantMessage("a1", "Trả lời đầu"),
  apiUserMessage("u2", "Câu hỏi thứ hai"),
].map(toPharmaMessage)

function send() {
  return createChatTransport().sendMessages({
    trigger: "submit-message",
    chatId: "c1",
    messageId: undefined,
    messages: history,
    abortSignal: undefined,
  })
}

describe("createChatTransport", () => {
  afterEach(() => {
    document.cookie = "csrftoken=; max-age=0; path=/"
  })

  test("posts only the conversation id and the last question with the CSRF header", async () => {
    document.cookie = "csrftoken=token-123; path=/"
    const captured: { body: unknown; csrf: string | null }[] = []
    worker.use(
      http.post(`*${CHAT_STREAM_API}`, async ({ request }) => {
        captured.push({
          body: await request.json(),
          csrf: request.headers.get("x-csrftoken"),
        })
        const chunks: UIMessageChunk[] = [
          { type: "start", messageId: "a2" },
          { type: "finish", finishReason: "stop" },
        ]
        return createUIMessageStreamResponse({
          stream: new ReadableStream<UIMessageChunk>({
            start(controller) {
              for (const chunk of chunks) {
                controller.enqueue(chunk)
              }
              controller.close()
            },
          }),
        })
      })
    )

    const chunks = await readAll(await send())

    expect(captured).toEqual([
      {
        body: { conversation_id: "c1", message: "Câu hỏi thứ hai" },
        csrf: "token-123",
      },
    ])
    expect(chunks.map((chunk) => chunk.type)).toEqual(["start", "finish"])
  })

  test("rejects with the problem body when the stream cannot start", async () => {
    worker.use(
      http.post(`*${CHAT_STREAM_API}`, () =>
        HttpResponse.json(
          {
            type: "urn:pharma-agent:problem:agent-unavailable",
            title: "Agent unavailable",
            status: 503,
            code: "AGENT_UNAVAILABLE",
          },
          {
            status: 503,
            headers: { "Content-Type": "application/problem+json" },
          }
        )
      )
    )
    const error = await send().then(
      () => new Error("expected a rejection"),
      (reason: unknown) =>
        reason instanceof Error ? reason : new Error("not an Error")
    )
    expect(readStreamProblem(error)?.code).toBe("AGENT_UNAVAILABLE")
  })
})
