import { type InfiniteData, QueryClient } from "@tanstack/react-query"
import { describe, expect, test } from "vitest"

import {
  getListConversationsInfiniteQueryKey,
  getListMessagesInfiniteQueryKey,
} from "~/api/gen/endpoints"
import type { ConversationPage, MessagePage } from "~/api/gen/schemas"

import {
  apiAssistantMessage,
  conversationPage,
  conversationView,
  messagePage,
} from "../../../../tests/chat/fixtures"

import {
  removeConversationFromCache,
  replaceConversationInCache,
  setMessageFeedbackInCache,
  updateConversationTitle,
} from "./conversation-cache"

type Cursor = string | null | undefined

const LIST_KEY = getListConversationsInfiniteQueryKey({ limit: 20 })

function seedConversations(queryClient: QueryClient) {
  queryClient.setQueryData<InfiniteData<ConversationPage, Cursor>>(LIST_KEY, {
    pages: [
      conversationPage(
        [
          conversationView("c1", "Cuộc trò chuyện mới"),
          conversationView("c2", "Ibuprofen"),
        ],
        "cursor-1"
      ),
      conversationPage([conversationView("c3", "Amoxicillin")]),
    ],
    pageParams: [undefined, "cursor-1"],
  })
}

function titles(queryClient: QueryClient): string[] {
  const data =
    queryClient.getQueryData<InfiniteData<ConversationPage, Cursor>>(LIST_KEY)
  return (
    data?.pages.flatMap((page) => page.items.map((item) => item.title)) ?? []
  )
}

describe("conversation cache", () => {
  test("updates a title and reports whether it was cached", () => {
    const queryClient = new QueryClient()
    seedConversations(queryClient)
    expect(
      updateConversationTitle(queryClient, {
        id: "c1",
        title: "Liều paracetamol",
      })
    ).toBe(true)
    expect(
      updateConversationTitle(queryClient, { id: "missing", title: "x" })
    ).toBe(false)
    expect(titles(queryClient)).toEqual([
      "Liều paracetamol",
      "Ibuprofen",
      "Amoxicillin",
    ])
  })

  test("replaces and removes conversations across pages", () => {
    const queryClient = new QueryClient()
    seedConversations(queryClient)
    replaceConversationInCache(
      queryClient,
      conversationView("c3", "Amoxicillin cho trẻ")
    )
    removeConversationFromCache(queryClient, "c2")
    expect(titles(queryClient)).toEqual([
      "Cuộc trò chuyện mới",
      "Amoxicillin cho trẻ",
    ])
  })

  test("sets feedback on a cached history message", () => {
    const queryClient = new QueryClient()
    const key = getListMessagesInfiniteQueryKey("c1", { limit: 30 })
    queryClient.setQueryData<InfiniteData<MessagePage, Cursor>>(key, {
      pages: [
        messagePage([
          apiAssistantMessage("a1", "Liều."),
          apiAssistantMessage("a2", "Khác."),
        ]),
      ],
      pageParams: [undefined],
    })
    setMessageFeedbackInCache(queryClient, "c1", "a1", {
      rating: "down",
      note: "Thiếu",
    })
    const items =
      queryClient.getQueryData<InfiniteData<MessagePage, Cursor>>(key)?.pages[0]
        ?.items ?? []
    expect(items.map((item) => item.metadata.feedback ?? null)).toEqual([
      { rating: "down", note: "Thiếu" },
      null,
    ])
  })
})
