import { useCallback, useEffect, useMemo } from "react"
import { useTranslation } from "react-i18next"
import { useLocation, useNavigate } from "react-router"

import { useListMessagesInfinite } from "~/api/gen/endpoints"
import { isApiError } from "~/api/problem"
import { Button } from "~/components/ui/button"
import {
  Empty,
  EmptyContent,
  EmptyHeader,
  EmptyTitle,
} from "~/components/ui/empty"
import { Spinner } from "~/components/ui/spinner"
import { toast } from "~/components/ui/toast"
import { apiErrorMessage } from "~/i18n/error-message"

import { ChatSession } from "./components/ChatSession"
import { NewChat } from "./components/NewChat"
import { toPharmaMessage } from "./lib/chat-messages"
import { readInitialQuestion } from "./lib/location-state"

export const HISTORY_PAGE_SIZE = 30

export type ChatThreadProps = {
  conversationId: string | undefined
}

export function ChatThreadFallback() {
  const { t } = useTranslation("chat")
  return (
    <output className="flex h-full items-center justify-center gap-2 text-sm text-muted-foreground">
      <Spinner />
      {t("thread.loading")}
    </output>
  )
}

function ConversationChat({ conversationId }: { conversationId: string }) {
  const { t } = useTranslation("chat")
  const { t: tErrors } = useTranslation("errors")
  const navigate = useNavigate()
  const location = useLocation()
  const initialQuestion = readInitialQuestion(location.state)
  const {
    data,
    error,
    isError,
    hasNextPage,
    isFetchingNextPage,
    fetchNextPage,
    refetch,
  } = useListMessagesInfinite(
    conversationId,
    { limit: HISTORY_PAGE_SIZE },
    {
      query: {
        initialPageParam: undefined,
        getNextPageParam: (page) => page.next_cursor ?? undefined,
        staleTime: Number.POSITIVE_INFINITY,
        gcTime: 0,
        refetchOnWindowFocus: false,
        refetchOnReconnect: false,
      },
    }
  )
  const notFound = isApiError(error) && error.code === "CONVERSATION_NOT_FOUND"
  const firstPage = data?.pages[0]
  const initialMessages = useMemo(
    () => firstPage?.items.map(toPharmaMessage) ?? [],
    [firstPage]
  )

  useEffect(() => {
    if (!notFound) {
      return
    }
    toast.add({ title: apiErrorMessage(error, tErrors), type: "error" })
    void navigate("/chat", { replace: true })
  }, [notFound, error, navigate, tErrors])

  const clearInitialQuestion = useCallback(() => {
    void navigate(location.pathname, { replace: true, state: null })
  }, [navigate, location.pathname])

  const loadOlder = useCallback(() => {
    void fetchNextPage()
  }, [fetchNextPage])

  if (isError && data === undefined) {
    if (notFound) {
      return null
    }
    return (
      <Empty className="h-full">
        <EmptyHeader>
          <EmptyTitle>{t("thread.loadFailed")}</EmptyTitle>
        </EmptyHeader>
        <EmptyContent>
          <Button variant="outline" onClick={() => void refetch()}>
            {t("thread.reload")}
          </Button>
        </EmptyContent>
      </Empty>
    )
  }

  if (data === undefined || firstPage === undefined) {
    return <ChatThreadFallback />
  }

  return (
    <ChatSession
      conversationId={conversationId}
      initialMessages={initialMessages}
      pages={data.pages}
      hasOlder={hasNextPage}
      loadingOlder={isFetchingNextPage}
      onLoadOlder={loadOlder}
      initialQuestion={initialQuestion}
      onInitialQuestionSent={clearInitialQuestion}
    />
  )
}

export function ChatThread({ conversationId }: ChatThreadProps) {
  if (conversationId === undefined) {
    return <NewChat />
  }
  return (
    <ConversationChat key={conversationId} conversationId={conversationId} />
  )
}
