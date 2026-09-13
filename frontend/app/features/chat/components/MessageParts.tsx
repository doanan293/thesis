import { type ReactNode, useMemo } from "react"
import { useTranslation } from "react-i18next"

import { AgentStatus } from "~/components/elements/agent-status"
import { ErrorState } from "~/components/elements/error-state"
import { GuardrailNotice } from "~/components/elements/guardrail-notice"
import { Badge } from "~/components/ui/badge"
import { Bubble, BubbleContent } from "~/components/ui/bubble"
import { Marker, MarkerContent } from "~/components/ui/marker"
import { Message, MessageContent, MessageFooter } from "~/components/ui/message"
import {
  type MessageCitations,
  MessageCitationsProvider,
} from "~/features/citations/citation-context"

import type { PharmaUIMessage } from "../lib/message-schema"
import {
  describeMessage,
  type MessageView,
  PHASE_KEYS,
} from "../lib/message-view"

import { AnswerMarkdown } from "./AnswerMarkdown"
import { EvidencePanel } from "./EvidencePanel"
import { SourcesList } from "./SourcesList"

export type MessagePartsProps = {
  message: PharmaUIMessage
  streaming: boolean
  stopped: boolean
  failed: boolean
  retrying: boolean
  onRetry: (messageId: string) => void
  onOpenCitation: (messageId: string, index: number) => void
  renderActions?: (message: PharmaUIMessage, view: MessageView) => ReactNode
}

export function MessageParts({
  message,
  streaming,
  stopped,
  failed,
  retrying,
  onRetry,
  onOpenCitation,
  renderActions,
}: MessagePartsProps) {
  const { t } = useTranslation("chat")
  const view = useMemo(
    () => describeMessage(message, { streaming, failed }),
    [message, streaming, failed]
  )
  const citations = useMemo<MessageCitations>(
    () => ({
      messageId: message.id,
      sources: new Map(view.sources.map((source) => [source.index, source])),
      openCitation: (index) => onOpenCitation(message.id, index),
    }),
    [message.id, view.sources, onOpenCitation]
  )

  if (message.role === "user") {
    return (
      <Message align="end">
        <MessageContent>
          <Bubble variant="secondary" align="end">
            <BubbleContent className="whitespace-pre-wrap">
              {view.text}
            </BubbleContent>
          </Bubble>
        </MessageContent>
      </Message>
    )
  }

  let phaseLabel = ""
  if (view.phase !== undefined) {
    phaseLabel =
      view.phase.phase === "searching" && typeof view.phase.round === "number"
        ? t("phase.searchingRound", { round: view.phase.round })
        : t(PHASE_KEYS[view.phase.phase])
  }

  let body: ReactNode = null
  if (view.notice?.kind === "guardrail") {
    body = (
      <GuardrailNotice
        title={
          view.notice.reason === "blocked"
            ? t("guardrail.blocked")
            : t("guardrail.redirected")
        }
      >
        <AnswerMarkdown text={view.text} streaming={false} />
      </GuardrailNotice>
    )
  } else if (view.notice?.kind === "error") {
    const detail =
      view.notice.reason === "timeout"
        ? t("error.timeout")
        : view.notice.reason === "network"
          ? t("error.network")
          : t("error.failed")
    body = (
      <>
        {view.hasText ? (
          <AnswerMarkdown text={view.text} streaming={false} />
        ) : null}
        <ErrorState
          title={t("error.title")}
          detail={detail}
          retrying={retrying}
          onRetry={() => onRetry(message.id)}
        />
      </>
    )
  } else if (view.hasText) {
    body = <AnswerMarkdown text={view.text} streaming={streaming} />
  }

  return (
    <Message align="start">
      <MessageContent>
        {view.showStatus ? (
          <AgentStatus state="working" label={phaseLabel} />
        ) : null}
        {view.skills.length > 0 ? (
          <ul
            aria-label={t("skills.label")}
            className="flex flex-wrap items-center gap-1.5"
          >
            {view.skills.map((skill) => (
              <li key={skill.name}>
                <Badge variant="secondary">{skill.title}</Badge>
              </li>
            ))}
          </ul>
        ) : null}
        {view.evidence.length > 0 ? (
          <EvidencePanel
            items={view.evidence}
            searching={streaming && !view.hasText}
          />
        ) : null}
        <MessageCitationsProvider value={citations}>
          {body}
        </MessageCitationsProvider>
        {view.sources.length > 0 && view.notice === null ? (
          <SourcesList
            sources={view.sources}
            onSelect={citations.openCitation}
          />
        ) : null}
        {stopped ? (
          <Marker variant="separator">
            <MarkerContent>{t("stopped")}</MarkerContent>
          </Marker>
        ) : null}
        {renderActions !== undefined && view.canGiveFeedback ? (
          <MessageFooter>{renderActions(message, view)}</MessageFooter>
        ) : null}
      </MessageContent>
    </Message>
  )
}
