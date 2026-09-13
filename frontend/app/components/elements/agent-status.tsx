import { CheckIcon } from "lucide-react"
import type { ComponentProps } from "react"
import { useTranslation } from "react-i18next"

import { cn } from "~/lib/utils"

import { mono, paper } from "./surfaces"

export type AgentState = "working" | "waiting" | "done"

export type AgentStatusProps = Omit<ComponentProps<"output">, "children"> & {
  state: AgentState
  label: string
  elapsed?: string
}

export function AgentStatus({
  state,
  label,
  elapsed,
  className,
  ...props
}: AgentStatusProps) {
  const { t } = useTranslation("chat")
  return (
    <output
      data-slot="agent-status"
      className={cn(
        paper,
        "flex w-fit items-center gap-2.5 rounded-full px-3.5 py-1.5",
        className
      )}
      {...props}
    >
      {state === "done" ? (
        <CheckIcon aria-hidden className="size-3 shrink-0 text-emerald-500" />
      ) : (
        <span
          aria-hidden
          className={cn(
            "size-1.5 shrink-0 rounded-full motion-reduce:animate-none",
            state === "working"
              ? "animate-pulse bg-blue-500 dark:bg-blue-400"
              : "border border-foreground/35"
          )}
        />
      )}
      <span className="sr-only">
        {state === "done" ? t("status.done") : t("status.working")}
      </span>
      <span
        key={label}
        className="max-w-72 animate-in truncate text-xs duration-300 blur-in-[2px] fade-in motion-reduce:animate-none"
      >
        {label}
      </span>
      {elapsed !== undefined && state !== "done" ? (
        <span className={cn(mono, "text-foreground/30 tabular-nums")}>
          {elapsed}
        </span>
      ) : null}
    </output>
  )
}
