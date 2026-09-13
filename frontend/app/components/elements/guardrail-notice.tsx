import { ShieldIcon } from "lucide-react"
import type { ReactNode } from "react"

import { cn } from "~/lib/utils"

import { paper } from "./surfaces"

export type GuardrailNoticeProps = {
  title: string
  children: ReactNode
  className?: string
}

export function GuardrailNotice({
  title,
  children,
  className,
}: GuardrailNoticeProps) {
  return (
    <div
      data-slot="guardrail-notice"
      className={cn(
        paper,
        "flex w-full max-w-xl flex-col gap-3 rounded-[20px] p-4",
        className
      )}
    >
      <div className="flex items-center gap-2.5">
        <span className="flex size-7 shrink-0 items-center justify-center rounded-lg bg-amber-500/12 text-amber-600 dark:text-amber-400">
          <ShieldIcon aria-hidden className="size-3.5" />
        </span>
        <span className="min-w-0 flex-1 text-[13.5px] font-medium">
          {title}
        </span>
      </div>
      <div className="text-sm leading-relaxed text-foreground/70">
        {children}
      </div>
    </div>
  )
}
