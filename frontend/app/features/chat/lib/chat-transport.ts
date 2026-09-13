import { DefaultChatTransport } from "ai"

import type { ChatRequest } from "~/api/gen/schemas"
import { CSRF_HEADER, readCsrfToken } from "~/lib/csrf"

import { textOf } from "./chat-messages"
import type { PharmaUIMessage } from "./message-schema"

export const CHAT_STREAM_API = "/api/v1/chat/stream"

function csrfHeaders(): Record<string, string> {
  const token = readCsrfToken()
  return token === undefined ? {} : { [CSRF_HEADER]: token }
}

export function lastUserText(messages: readonly PharmaUIMessage[]): string {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]
    if (message?.role === "user") {
      return textOf(message)
    }
  }
  return ""
}

export function createChatTransport(): DefaultChatTransport<PharmaUIMessage> {
  return new DefaultChatTransport<PharmaUIMessage>({
    api: CHAT_STREAM_API,
    credentials: "same-origin",
    headers: csrfHeaders,
    prepareSendMessagesRequest: ({ id, messages }) => {
      const body: ChatRequest = {
        conversation_id: id,
        message: lastUserText(messages),
      }
      return { body }
    },
  })
}
