import { useChat } from "@ai-sdk/react"
import { useQueryClient } from "@tanstack/react-query"
import { ArrowDownIcon } from "lucide-react"
import { useCallback, useEffect, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { useLocation, useNavigate } from "react-router"

import type { MessageFeedback, MessagePage } from "~/api/gen/schemas"
import { ErrorState } from "~/components/elements/error-state"
import {
  MessageScroller,
  MessageScrollerButton,
  MessageScrollerContent,
  MessageScrollerItem,
  MessageScrollerProvider,
  MessageScrollerViewport,
} from "~/components/ui/message-scroller"
import { toast } from "~/components/ui/toast"
import {
  type CitationTarget,
  CitationSheet,
} from "~/features/citations/CitationSheet"
import {
  invalidateConversations,
  updateConversationTitle,
} from "~/features/conversations/lib/conversation-cache"
import { apiErrorMessage } from "~/i18n/error-message"

import {
  dropTrailingTurn,
  olderMessagesFrom,
  prependOlder,
  questionBefore,
  textOf,
  withFeedback,
} from "../lib/chat-messages"
import { createChatTransport } from "../lib/chat-transport"
import {
  messageMetadataSchema,
  pharmaDataPartSchemas,
  type PharmaUIMessage,
} from "../lib/message-schema"
import type { MessageView } from "../lib/message-view"
import { readStreamProblem } from "../lib/stream-problem"

import { AgentUnavailableBanner } from "./AgentUnavailableBanner"
import { Composer } from "./Composer"
import { MessageActions } from "./MessageActions"
import { MessageParts } from "./MessageParts"
import { OlderMessagesLoader } from "./OlderMessagesLoader"

export type ChatSessionProps = {
  conversationId: string
  initialMessages: PharmaUIMessage[]
  pages: readonly MessagePage[]
  hasOlder: boolean
  loadingOlder: boolean
  onLoadOlder: () => void
  initialQuestion: string | undefined
  onInitialQuestionSent: () => void
}

const STANDALONE_ERROR_ID = "stream-error"

function withId(ids: ReadonlySet<string>, id: string): ReadonlySet<string> {
  const next = new Set(ids)
  next.add(id)
  return next
}

export function ChatSession({
  conversationId,
  initialMessages,
  pages,
  hasOlder,
  loadingOlder,
  onLoadOlder,
  initialQuestion,
  onInitialQuestionSent,
}: ChatSessionProps) {
  const { t } = useTranslation("chat")
  const { t: tErrors } = useTranslation("errors")
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const location = useLocation()
  const [transport] = useState(createChatTransport)
  const [draft, setDraft] = useState("")
  const [composerError, setComposerError] = useState<string | null>(null)
  const [agentUnavailable, setAgentUnavailable] = useState(false)
  const [stoppedIds, setStoppedIds] = useState<ReadonlySet<string>>(
    () => new Set()
  )
  const [failedIds, setFailedIds] = useState<ReadonlySet<string>>(
    () => new Set()
  )
  const [retryingId, setRetryingId] = useState<string | null>(null)
  const [citation, setCitation] = useState<CitationTarget | null>(null)
  const [userScrolled, setUserScrolled] = useState(false)
  const [contentFits, setContentFits] = useState(false)
  const viewportRef = useRef<HTMLDivElement>(null)
  const contentRef = useRef<HTMLDivElement>(null)
  const lastQuestionRef = useRef("")
  const appliedPagesRef = useRef(1)
  const initialSentRef = useRef(false)

  const { messages, sendMessage, status, stop, setMessages, error } =
    useChat<PharmaUIMessage>({
      id: conversationId,
      messages: initialMessages,
      transport,
      dataPartSchemas: pharmaDataPartSchemas,
      messageMetadataSchema,
      onData: (part) => {
        if (
          part.type === "data-conversation" &&
          !updateConversationTitle(queryClient, part.data)
        ) {
          void invalidateConversations(queryClient)
        }
      },
      onFinish: ({ message, isAbort, isError, isDisconnect }) => {
        setRetryingId(null)
        if (isAbort) {
          setStoppedIds((current) => withId(current, message.id))
        }
        if (isError || isDisconnect) {
          setFailedIds((current) => withId(current, message.id))
        }
        if (message.metadata?.persisted === false) {
          toast.add({ title: t("notPersisted"), type: "warning" })
        }
        void invalidateConversations(queryClient)
      },
      onError: (streamError) => {
        const problem = readStreamProblem(streamError)
        if (problem === null) {
          return
        }
        setMessages((current) => dropTrailingTurn(current))
        setDraft(lastQuestionRef.current)
        if (problem.status === 401) {
          void navigate(
            `/login?next=${encodeURIComponent(location.pathname + location.search)}`
          )
        } else if (problem.code === "CONVERSATION_NOT_FOUND") {
          toast.add({ title: apiErrorMessage(problem, tErrors), type: "error" })
          void navigate("/chat", { replace: true })
        } else if (problem.status === 503) {
          setAgentUnavailable(true)
        } else {
          setComposerError(problem.detail ?? apiErrorMessage(problem, tErrors))
        }
      },
    })

  const busy = status === "submitted" || status === "streaming"
  const lastMessage = messages.at(-1)
  const streamingId =
    busy && lastMessage?.role === "assistant" ? lastMessage.id : undefined
  const streamProblem = error === undefined ? null : readStreamProblem(error)
  const showStandaloneError =
    status === "error" && streamProblem === null && lastMessage?.role === "user"

  const ask = useCallback(
    (question: string) => {
      lastQuestionRef.current = question
      setComposerError(null)
      setAgentUnavailable(false)
      void sendMessage({ text: question })
    },
    [sendMessage]
  )

  const retry = useCallback(
    (messageId: string) => {
      const question = questionBefore(messages, messageId)
      if (question === undefined) {
        return
      }
      setRetryingId(messageId)
      if (failedIds.has(messageId) && messages.at(-1)?.id === messageId) {
        setMessages(dropTrailingTurn(messages))
      }
      ask(question)
    },
    [messages, failedIds, setMessages, ask]
  )

  const retryLastQuestion = useCallback(() => {
    const last = messages.at(-1)
    if (last?.role !== "user") {
      return
    }
    setMessages(messages.slice(0, -1))
    ask(textOf(last))
  }, [messages, setMessages, ask])

  const changeFeedback = useCallback(
    (messageId: string, feedback: MessageFeedback | null) => {
      setMessages((current) => withFeedback(current, messageId, feedback))
    },
    [setMessages]
  )

  const renderActions = useCallback(
    (message: PharmaUIMessage, view: MessageView) => (
      <MessageActions
        conversationId={conversationId}
        messageId={message.id}
        text={view.text}
        feedback={view.feedback}
        onFeedbackChange={changeFeedback}
      />
    ),
    [conversationId, changeFeedback]
  )

  const openCitation = useCallback((messageId: string, index: number) => {
    setCitation({ messageId, index })
  }, [])

  const markUserScrolled = useCallback(() => setUserScrolled(true), [])

  useEffect(() => {
    const viewport = viewportRef.current
    const content = contentRef.current
    if (viewport === null || content === null) {
      return undefined
    }
    const observer = new ResizeObserver(() => {
      setContentFits(viewport.scrollHeight <= viewport.clientHeight + 1)
    })
    observer.observe(viewport)
    observer.observe(content)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    if (pages.length <= appliedPagesRef.current) {
      return
    }
    const older = olderMessagesFrom(pages, appliedPagesRef.current)
    appliedPagesRef.current = pages.length
    setMessages((current) => prependOlder(current, older))
  }, [pages, setMessages])

  useEffect(() => {
    if (initialQuestion === undefined || initialSentRef.current) {
      return
    }
    initialSentRef.current = true
    lastQuestionRef.current = initialQuestion
    void sendMessage({ text: initialQuestion })
    onInitialQuestionSent()
  }, [initialQuestion, sendMessage, onInitialQuestionSent])

  return (
    <div className="flex h-full min-h-0 flex-col">
      {agentUnavailable ? <AgentUnavailableBanner /> : null}
      <MessageScrollerProvider autoScroll defaultScrollPosition="last-anchor">
        <MessageScroller className="min-h-0 flex-1">
          <OlderMessagesLoader
            hasOlder={hasOlder}
            loadingOlder={loadingOlder}
            userScrolled={userScrolled}
            contentFits={contentFits}
            onLoadOlder={onLoadOlder}
          />
          <MessageScrollerViewport
            ref={viewportRef}
            aria-label={t("thread.label")}
            onWheel={markUserScrolled}
            onTouchMove={markUserScrolled}
            onKeyDown={markUserScrolled}
          >
            <MessageScrollerContent
              ref={contentRef}
              className="mx-auto w-full max-w-3xl px-4 py-6"
            >
              {messages.map((message) => (
                <MessageScrollerItem
                  key={message.id}
                  messageId={message.id}
                  data-message-id={message.id}
                  scrollAnchor={message.role === "user"}
                >
                  <MessageParts
                    message={message}
                    streaming={message.id === streamingId}
                    stopped={stoppedIds.has(message.id)}
                    failed={failedIds.has(message.id)}
                    retrying={retryingId === message.id}
                    onRetry={retry}
                    onOpenCitation={openCitation}
                    renderActions={renderActions}
                  />
                </MessageScrollerItem>
              ))}
              {showStandaloneError ? (
                <MessageScrollerItem messageId={STANDALONE_ERROR_ID}>
                  <ErrorState
                    title={t("error.title")}
                    detail={t("error.network")}
                    retrying={false}
                    onRetry={retryLastQuestion}
                  />
                </MessageScrollerItem>
              ) : null}
            </MessageScrollerContent>
          </MessageScrollerViewport>
          <MessageScrollerButton direction="end">
            <ArrowDownIcon aria-hidden />
            <span className="sr-only">{t("thread.scrollToEnd")}</span>
          </MessageScrollerButton>
        </MessageScroller>
      </MessageScrollerProvider>
      <Composer
        value={draft}
        onValueChange={(value) => {
          setDraft(value)
          setComposerError(null)
        }}
        busy={busy}
        error={composerError}
        onSubmit={(question) => {
          setDraft("")
          ask(question)
        }}
        onStop={() => void stop()}
      />
      <CitationSheet target={citation} onClose={() => setCitation(null)} />
    </div>
  )
}
