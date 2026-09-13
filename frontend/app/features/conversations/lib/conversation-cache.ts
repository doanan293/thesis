import type { InfiniteData, QueryClient } from "@tanstack/react-query"

import {
  getListConversationsInfiniteQueryKey,
  getListMessagesInfiniteQueryKey,
} from "~/api/gen/endpoints"
import type {
  ConversationPage,
  ConversationView,
  MessageFeedback,
  MessagePage,
} from "~/api/gen/schemas"

type Cursor = string | null | undefined

function mapConversationItems(
  queryClient: QueryClient,
  update: (items: ConversationView[]) => ConversationView[]
): void {
  queryClient.setQueriesData<InfiniteData<ConversationPage, Cursor>>(
    { queryKey: getListConversationsInfiniteQueryKey() },
    (data) =>
      data === undefined
        ? data
        : {
            ...data,
            pages: data.pages.map((page) => ({
              ...page,
              items: update(page.items),
            })),
          }
  )
}

export function updateConversationTitle(
  queryClient: QueryClient,
  conversation: { id: string; title: string }
): boolean {
  let found = false
  mapConversationItems(queryClient, (items) =>
    items.map((item) => {
      if (item.id !== conversation.id) {
        return item
      }
      found = true
      return { ...item, title: conversation.title }
    })
  )
  return found
}

export function replaceConversationInCache(
  queryClient: QueryClient,
  conversation: ConversationView
): void {
  mapConversationItems(queryClient, (items) =>
    items.map((item) => (item.id === conversation.id ? conversation : item))
  )
}

export function removeConversationFromCache(
  queryClient: QueryClient,
  conversationId: string
): void {
  mapConversationItems(queryClient, (items) =>
    items.filter((item) => item.id !== conversationId)
  )
}

export function invalidateConversations(
  queryClient: QueryClient
): Promise<void> {
  return queryClient.invalidateQueries({
    queryKey: getListConversationsInfiniteQueryKey(),
  })
}

export function setMessageFeedbackInCache(
  queryClient: QueryClient,
  conversationId: string,
  messageId: string,
  feedback: MessageFeedback | null
): void {
  queryClient.setQueriesData<InfiniteData<MessagePage, Cursor>>(
    { queryKey: getListMessagesInfiniteQueryKey(conversationId) },
    (data) =>
      data === undefined
        ? data
        : {
            ...data,
            pages: data.pages.map((page) => ({
              ...page,
              items: page.items.map((item) =>
                item.id === messageId
                  ? { ...item, metadata: { ...item.metadata, feedback } }
                  : item
              ),
            })),
          }
  )
}
