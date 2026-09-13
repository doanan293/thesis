import { CheckIcon, ThumbsDownIcon } from "lucide-react"
import { useTranslation } from "react-i18next"

import { cn } from "~/lib/utils"

import { field, inkButton, mono } from "./surfaces"

export interface FeedbackReason {
  id: string
  label: string
}

export type FeedbackDialogProps = {
  reasons: readonly FeedbackReason[]
  selected: readonly string[]
  note: string
  sent: boolean
  onToggleReason: (id: string) => void
  onNoteChange: (note: string) => void
  onSubmit: () => void
  className?: string
}

export function FeedbackDialog({
  reasons,
  selected,
  note,
  sent,
  onToggleReason,
  onNoteChange,
  onSubmit,
  className,
}: FeedbackDialogProps) {
  const { t } = useTranslation("chat")
  return (
    <div
      data-slot="feedback-dialog"
      className={cn(
        "flex w-full",
        sent ? "items-center gap-2.5 text-[13.5px]" : "flex-col gap-3",
        className
      )}
    >
      <output
        className={
          sent
            ? "flex animate-in items-center gap-2.5 duration-300 fade-in"
            : "sr-only"
        }
      >
        {sent ? (
          <>
            <CheckIcon
              aria-hidden
              className="size-4 shrink-0 text-emerald-500"
            />
            {t("feedback.sent")}
          </>
        ) : null}
      </output>

      {sent ? null : (
        <>
          <div className="flex items-center gap-2.5">
            <span className="flex size-7 shrink-0 items-center justify-center rounded-lg bg-foreground/[0.05] text-foreground/45">
              <ThumbsDownIcon aria-hidden className="size-3.5" />
            </span>
            <span className="text-[13.5px] font-medium">
              {t("feedback.title")}
            </span>
            <span className={cn(mono, "ms-auto text-foreground/30")}>
              {t("feedback.optional")}
            </span>
          </div>

          <div className="flex flex-wrap gap-1.5">
            {reasons.map((reason) => {
              const active = selected.includes(reason.id)
              return (
                <button
                  key={reason.id}
                  type="button"
                  aria-pressed={active}
                  onClick={() => onToggleReason(reason.id)}
                  className={cn(
                    "rounded-full px-2.5 py-1 text-xs transition-[background-color,color,scale] duration-150 active:scale-[0.96]",
                    active
                      ? "bg-foreground text-background"
                      : cn(field, "text-foreground/55 hover:text-foreground/90")
                  )}
                >
                  {reason.label}
                </button>
              )
            })}
          </div>

          <textarea
            value={note}
            onChange={(event) => onNoteChange(event.target.value)}
            rows={3}
            placeholder={t("feedback.notePlaceholder")}
            aria-label={t("feedback.notePlaceholder")}
            className={cn(
              field,
              "resize-none rounded-xl px-3 py-2 text-xs text-foreground/80 outline-none placeholder:text-foreground/30 focus-visible:ring-1 focus-visible:ring-foreground/20"
            )}
          />

          <button
            type="button"
            onClick={onSubmit}
            className={cn(
              inkButton,
              "flex h-8 items-center justify-center self-end rounded-full px-3.5 text-xs font-medium"
            )}
          >
            {t("feedback.submit")}
          </button>
        </>
      )}
    </div>
  )
}
