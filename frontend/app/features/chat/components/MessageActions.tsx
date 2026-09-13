import { useQueryClient } from "@tanstack/react-query"
import { CheckIcon, CopyIcon, ThumbsDownIcon, ThumbsUpIcon } from "lucide-react"
import { useEffect, useMemo, useState } from "react"
import { useTranslation } from "react-i18next"

import { useSubmitFeedback } from "~/api/gen/endpoints"
import type { MessageFeedback } from "~/api/gen/schemas"
import {
  FeedbackDialog,
  type FeedbackReason,
} from "~/components/elements/feedback-dialog"
import { Button } from "~/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "~/components/ui/dialog"
import { toast } from "~/components/ui/toast"
import { setMessageFeedbackInCache } from "~/features/conversations/lib/conversation-cache"
import { cn } from "~/lib/utils"

import { composeFeedbackNote } from "../lib/feedback-note"

export type MessageActionsProps = {
  conversationId: string
  messageId: string
  text: string
  feedback: MessageFeedback | null
  onFeedbackChange: (
    messageId: string,
    feedback: MessageFeedback | null
  ) => void
}

const COPIED_RESET_MS = 2000

export function MessageActions({
  conversationId,
  messageId,
  text,
  feedback,
  onFeedbackChange,
}: MessageActionsProps) {
  const { t } = useTranslation("chat")
  const queryClient = useQueryClient()
  const submitFeedback = useSubmitFeedback()
  const [copied, setCopied] = useState(false)
  const [dialogOpen, setDialogOpen] = useState(false)
  const [selected, setSelected] = useState<string[]>([])
  const [note, setNote] = useState("")
  const [sent, setSent] = useState(false)

  const reasons = useMemo<FeedbackReason[]>(
    () => [
      { id: "incorrect", label: t("feedback.reasons.incorrect") },
      { id: "incomplete", label: t("feedback.reasons.incomplete") },
      { id: "wrongSource", label: t("feedback.reasons.wrongSource") },
      { id: "unclear", label: t("feedback.reasons.unclear") },
    ],
    [t]
  )

  useEffect(() => {
    if (!copied) {
      return undefined
    }
    const timer = window.setTimeout(() => setCopied(false), COPIED_RESET_MS)
    return () => window.clearTimeout(timer)
  }, [copied])

  function apply(next: MessageFeedback) {
    const previous = feedback
    onFeedbackChange(messageId, next)
    setMessageFeedbackInCache(queryClient, conversationId, messageId, next)
    submitFeedback.mutate(
      { messageId, data: next },
      {
        onError: () => {
          onFeedbackChange(messageId, previous)
          setMessageFeedbackInCache(
            queryClient,
            conversationId,
            messageId,
            previous
          )
          toast.add({ title: t("actions.feedbackFailed"), type: "error" })
        },
      }
    )
  }

  async function copy() {
    await navigator.clipboard.writeText(text)
    setCopied(true)
  }

  function openNotHelpful() {
    setSelected([])
    setNote("")
    setSent(false)
    setDialogOpen(true)
    apply({ rating: "down", note: "" })
  }

  function toggleReason(id: string) {
    setSelected((current) =>
      current.includes(id)
        ? current.filter((value) => value !== id)
        : [...current, id]
    )
  }

  function sendDetails() {
    const labels = reasons
      .filter((reason) => selected.includes(reason.id))
      .map((reason) => reason.label)
    apply({ rating: "down", note: composeFeedbackNote(labels, note) })
    setSent(true)
  }

  return (
    <div className="flex items-center gap-0.5">
      <Button
        variant="ghost"
        size="icon-sm"
        aria-label={copied ? t("actions.copied") : t("actions.copy")}
        onClick={() => void copy()}
      >
        {copied ? <CheckIcon aria-hidden /> : <CopyIcon aria-hidden />}
      </Button>
      <Button
        variant="ghost"
        size="icon-sm"
        aria-label={t("actions.up")}
        aria-pressed={feedback?.rating === "up"}
        onClick={() => apply({ rating: "up", note: "" })}
      >
        <ThumbsUpIcon
          aria-hidden
          className={cn(feedback?.rating === "up" && "fill-current")}
        />
      </Button>
      <Button
        variant="ghost"
        size="icon-sm"
        aria-label={t("actions.down")}
        aria-pressed={feedback?.rating === "down"}
        onClick={openNotHelpful}
      >
        <ThumbsDownIcon
          aria-hidden
          className={cn(feedback?.rating === "down" && "fill-current")}
        />
      </Button>
      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader className="sr-only">
            <DialogTitle>{t("feedback.title")}</DialogTitle>
          </DialogHeader>
          <FeedbackDialog
            reasons={reasons}
            selected={selected}
            note={note}
            sent={sent}
            onToggleReason={toggleReason}
            onNoteChange={setNote}
            onSubmit={sendDetails}
          />
        </DialogContent>
      </Dialog>
    </div>
  )
}
