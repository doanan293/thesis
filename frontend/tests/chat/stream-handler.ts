import { createChat } from "@shadcn/helpers/ai-sdk"
import { createUIMessageStreamResponse } from "ai"
import { type HttpHandler, http, HttpResponse } from "msw"
import { z } from "zod"

import type { MessageMetadata } from "~/api/gen/schemas"
import { CHAT_STREAM_API } from "~/features/chat/lib/chat-transport"
import type { PharmaUIMessage } from "~/features/chat/lib/message-schema"

import { CREATED_AT, pharmaSource } from "./fixtures"

const streamRequestSchema = z.object({
  conversation_id: z.string(),
  message: z.string(),
})

export type StreamRequest = z.infer<typeof streamRequestSchema>
export type ScriptedChat = ReturnType<typeof createChat<PharmaUIMessage>>

export const STREAM_URL = `*${CHAT_STREAM_API}`

export function finishMetadata(
  overrides: Partial<MessageMetadata> = {}
): MessageMetadata {
  return {
    status: "completed",
    createdAt: CREATED_AT,
    errorCode: null,
    usage: {
      llmCalls: 4,
      promptTokens: 900,
      completionTokens: 120,
      searchRounds: 1,
    },
    runId: "00000000000000000000000000000003",
    persisted: true,
    ...overrides,
  }
}

export type AnswerOptions = {
  id?: string
  conversationId?: string
  title?: string
  text?: string
  metadata?: Partial<MessageMetadata>
  slow?: boolean
  withSource?: boolean
}

export function answerChat(
  question: string,
  options: AnswerOptions = {}
): ScriptedChat {
  const text =
    options.text ??
    "Người lớn uống 0,5–1 g mỗi 4–6 giờ, tối đa 4 g mỗi ngày [1]."
  return createChat<PharmaUIMessage>()
    .user(question)
    .assistant(
      ({ writer }) => {
        writer.data({
          type: "data-conversation",
          data: {
            id: options.conversationId ?? "c1",
            title: options.title ?? "Cuộc trò chuyện mới",
          },
          transient: true,
        })
        writer.data({
          type: "data-phase",
          id: "phase",
          data: { phase: "searching", round: 1 },
        })
        if (options.slow === true) {
          writer.sleep(600)
        }
        writer.data({
          type: "data-phase",
          id: "phase",
          data: { phase: "answering" },
        })
        writer.data({
          type: "data-evidence",
          id: "evidence",
          data: {
            items: [
              {
                index: 1,
                source: "Dược thư Quốc gia Việt Nam (Xuất bản lần thứ 2)",
                title: "Paracetamol",
                section: "Liều lượng và cách dùng",
                startPage: 812,
                endPage: 813,
                snippet: "Người lớn và trẻ em trên 12 tuổi…",
              },
            ],
          },
        })
        if (text !== "") {
          if (options.slow === true) {
            writer.text(text, { delayMs: 150 })
          } else {
            writer.text(text)
          }
        }
        if (options.withSource !== false) {
          writer.sourceDocument({
            sourceId: "chunk-1",
            mediaType: "text/markdown",
            title: "Paracetamol › Liều lượng và cách dùng",
            providerMetadata: { pharma: { ...pharmaSource(1) } },
          })
        }
      },
      { id: options.id ?? "a-new", metadata: finishMetadata(options.metadata) }
    )
}

export function chatStreamHandler(
  chats: readonly ScriptedChat[],
  requests: StreamRequest[] = [],
  delayMs = 0
): HttpHandler {
  let served = 0
  return http.post(STREAM_URL, async ({ request }) => {
    const body = streamRequestSchema.parse(await request.json())
    requests.push(body)
    const chat = chats[Math.min(served, chats.length - 1)]
    served += 1
    if (chat === undefined) {
      return HttpResponse.error()
    }
    const stream = await chat.transport({ delayMs }).sendMessages({
      trigger: "submit-message",
      chatId: body.conversation_id,
      messageId: undefined,
      messages: chat.get(1),
      abortSignal: request.signal,
    })
    return createUIMessageStreamResponse({ stream })
  })
}

export function problemHandler(
  status: number,
  code: string,
  detail?: string
): HttpHandler {
  return http.post(STREAM_URL, () =>
    HttpResponse.json(
      {
        type: `urn:pharma-agent:problem:${code.toLowerCase().replaceAll("_", "-")}`,
        title: code,
        status,
        code,
        ...(detail === undefined ? {} : { detail }),
      },
      { status, headers: { "Content-Type": "application/problem+json" } }
    )
  )
}

export function networkErrorHandler(): HttpHandler {
  return http.post(STREAM_URL, () => HttpResponse.error())
}
