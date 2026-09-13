import { ChatThread, ChatThreadFallback } from "~/features/chat/ChatThread"

import type { Route } from "./+types/chat"

export function clientLoader({ params }: Route.ClientLoaderArgs) {
  return { conversationId: params.conversationId ?? null }
}

export function HydrateFallback() {
  return <ChatThreadFallback />
}

export default function ChatRoute({ loaderData }: Route.ComponentProps) {
  return (
    <div className="flex h-[calc(100svh-3rem)] min-h-0 flex-col">
      <ChatThread conversationId={loaderData.conversationId ?? undefined} />
    </div>
  )
}
