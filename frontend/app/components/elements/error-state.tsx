import { CircleAlertIcon, RefreshCwIcon } from "lucide-react"
import { useTranslation } from "react-i18next"

import { cn } from "~/lib/utils"

import { ShimmerLabel } from "./surfaces"

export type ErrorStateProps = {
  title: string
  detail: string
  retrying: boolean
  onRetry: () => void
  className?: string
}

export function ErrorState({
  title,
  detail,
  retrying,
  onRetry,
  className,
}: ErrorStateProps) {
  const { t } = useTranslation("chat")
  if (retrying) {
    return (
      <output
        data-slot="error-state"
        className={cn(
          "flex w-full max-w-xl animate-in items-center gap-2.5 text-sm duration-300 fade-in motion-reduce:animate-none",
          className
        )}
      >
        <RefreshCwIcon
          aria-hidden
          className="size-3.5 shrink-0 animate-spin text-foreground/45 motion-reduce:animate-none"
        />
        <ShimmerLabel className="relative inline-block text-foreground/55">
          {t("error.retrying")}
        </ShimmerLabel>
      </output>
    )
  }

  return (
    <div
      data-slot="error-state"
      role="alert"
      className={cn(
        "flex w-full max-w-xl animate-in items-start gap-2.5 rounded-2xl bg-red-500/[0.06] px-4 py-3 text-sm duration-300 fade-in motion-reduce:animate-none dark:bg-red-500/10",
        className
      )}
    >
      <CircleAlertIcon
        aria-hidden
        className="mt-0.5 size-4 shrink-0 text-red-500/80"
      />
      <div>
        <p className="font-medium text-red-600 dark:text-red-400">{title}</p>
        <p className="mt-0.5 text-[13px] leading-snug text-red-600/60 dark:text-red-400/60">
          {detail}
        </p>
      </div>
      <button
        type="button"
        onClick={onRetry}
        className="ms-auto flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium text-red-600 transition-colors hover:bg-red-500/10 dark:text-red-400"
      >
        <RefreshCwIcon aria-hidden className="size-3" />
        {t("error.retry")}
      </button>
    </div>
  )
}
