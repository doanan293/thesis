import { ChevronDownIcon } from "lucide-react"
import { useTranslation } from "react-i18next"

import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "~/components/ui/collapsible"
import { cn } from "~/lib/utils"

import { collapsePanel, fieldInteractive, mono, paper } from "./surfaces"

export interface SourceCard {
  index: number
  source: string
  title: string
  stale: boolean
}

export type SourcesProps = {
  sources: readonly SourceCard[]
  open: boolean
  onOpenChange: (open: boolean) => void
  onSelect: (index: number) => void
  className?: string
}

export function Sources({
  sources,
  open,
  onOpenChange,
  onSelect,
  className,
}: SourcesProps) {
  const { t } = useTranslation("chat")
  return (
    <Collapsible
      data-slot="sources"
      open={open}
      onOpenChange={onOpenChange}
      className={cn("w-full max-w-xl", className)}
    >
      <CollapsibleTrigger
        className={cn(
          fieldInteractive,
          "group/trigger inline-flex w-fit items-center gap-1.5 rounded-full px-3.5 py-2 text-xs text-foreground/60 outline-none hover:text-foreground/90"
        )}
      >
        <span>{t("sources.trigger", { count: sources.length })}</span>
        <ChevronDownIcon
          aria-hidden
          className="size-3 opacity-60 transition-transform duration-200 group-data-panel-open/trigger:rotate-180 motion-reduce:transition-none"
        />
      </CollapsibleTrigger>
      <CollapsibleContent className={cn(collapsePanel, "outline-none")}>
        <div className="grid grid-cols-1 gap-2 pt-2.5 sm:grid-cols-2">
          {sources.map((source) => (
            <button
              key={source.index}
              type="button"
              onClick={() => onSelect(source.index)}
              className={cn(
                paper,
                "flex flex-col gap-1.5 rounded-2xl p-3 text-start transition-transform hover:-translate-y-px"
              )}
            >
              <span className="flex items-center gap-1.5">
                <span className="flex h-4 min-w-4 shrink-0 items-center justify-center rounded bg-foreground/[0.06] px-1 text-[9px] font-medium text-foreground/45 tabular-nums">
                  {source.index}
                </span>
                <span className={cn(mono, "truncate text-foreground/40")}>
                  {source.source}
                </span>
              </span>
              <span className="line-clamp-2 text-[13px] leading-snug font-medium text-foreground/90">
                {source.title}
              </span>
              {source.stale ? (
                <span className="text-xs text-amber-600 dark:text-amber-400">
                  {t("sources.stale")}
                </span>
              ) : null}
            </button>
          ))}
        </div>
      </CollapsibleContent>
    </Collapsible>
  )
}
