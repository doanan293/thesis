import type {
  UIMessage as ApiUIMessage,
  MessageFeedback,
  MessagePage,
} from "~/api/gen/schemas"

import type { PharmaMessagePart, PharmaUIMessage } from "./message-schema"

export function toPharmaMessage(message: ApiUIMessage): PharmaUIMessage {
  return {
    id: message.id,
    role: message.role,
    metadata: message.metadata,
    parts: message.parts.map((part): PharmaMessagePart => {
      if (part.type === "text") {
        return { type: "text", text: part.text }
      }
      return {
        type: "source-document",
        sourceId: part.sourceId,
        mediaType: part.mediaType,
        title: part.title,
        providerMetadata: { pharma: { ...part.providerMetadata.pharma } },
      }
    }),
  }
}

export function prependOlder(
  current: readonly PharmaUIMessage[],
  older: readonly PharmaUIMessage[]
): PharmaUIMessage[] {
  const known = new Set(current.map((message) => message.id))
  return [...older.filter((message) => !known.has(message.id)), ...current]
}

export function olderMessagesFrom(
  pages: readonly MessagePage[],
  appliedPageCount: number
): PharmaUIMessage[] {
  return pages
    .slice(appliedPageCount)
    .toReversed()
    .flatMap((page) => page.items.map(toPharmaMessage))
}

export function textOf(message: PharmaUIMessage): string {
  return message.parts
    .flatMap((part) => (part.type === "text" ? [part.text] : []))
    .join("")
}

export function questionBefore(
  messages: readonly PharmaUIMessage[],
  assistantMessageId: string
): string | undefined {
  const position = messages.findIndex(
    (message) => message.id === assistantMessageId
  )
  for (let index = position - 1; index >= 0; index -= 1) {
    const message = messages[index]
    if (message?.role === "user") {
      return textOf(message)
    }
  }
  return undefined
}

export function withFeedback(
  messages: readonly PharmaUIMessage[],
  messageId: string,
  feedback: MessageFeedback | null
): PharmaUIMessage[] {
  return messages.map((message) =>
    message.id === messageId && message.metadata !== undefined
      ? { ...message, metadata: { ...message.metadata, feedback } }
      : message
  )
}

export function dropTrailingTurn(
  messages: readonly PharmaUIMessage[]
): PharmaUIMessage[] {
  const withoutAssistant =
    messages.at(-1)?.role === "assistant"
      ? messages.length - 1
      : messages.length
  const end =
    messages[withoutAssistant - 1]?.role === "user"
      ? withoutAssistant - 1
      : withoutAssistant
  return messages.slice(0, end)
}
