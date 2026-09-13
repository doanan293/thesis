import { createContext, type ReactNode, useContext } from "react"

import type { PharmaSourceMetadata } from "~/api/gen/schemas"

export type MessageCitations = {
  messageId: string
  sources: ReadonlyMap<number, PharmaSourceMetadata>
  openCitation: (index: number) => void
}

const MessageCitationsContext = createContext<MessageCitations | null>(null)

export function MessageCitationsProvider({
  value,
  children,
}: {
  value: MessageCitations
  children: ReactNode
}) {
  return (
    <MessageCitationsContext value={value}>{children}</MessageCitationsContext>
  )
}

export function useMessageCitations(): MessageCitations | null {
  return useContext(MessageCitationsContext)
}
