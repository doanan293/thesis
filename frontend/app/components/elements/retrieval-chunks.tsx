import { ChevronDownIcon, DatabaseIcon } from "lucide-react"
import { useTranslation } from "react-i18next"

import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "~/components/ui/collapsible"
import { cn } from "~/lib/utils"

import {
  collapsePanel,
  fieldInteractive,
  mono,
  paper,
  ShimmerLabel,
} from "./surfaces"

export interface RetrievalChunk {
  id: string
  source: string
  locator: string
  text: string
}

export type RetrievalChunksProps = {
  chunks: readonly RetrievalChunk[]
  searching: boolean
  open: boolean
  onOpenChange: (open: boolean) => void
  className?: string
}

export function RetrievalChunks({
  chunks,
  searching,
  open,
  onOpenChange,
  className,
}: RetrievalChunksProps) {
  const { t } = useTranslation("chat")
  return (
    <Collapsible
      data-slot="retrieval-chunks"
      open={open}
      onOpenChange={onOpenChange}
      className={cn("flex w-full max-w-xl flex-col", className)}
    >
      <CollapsibleTrigger
        className={cn(
          fieldInteractive,
          "group/trigger inline-flex w-fit items-center gap-1.5 rounded-full px-3.5 py-2 text-xs text-foreground/70 outline-none"
        )}
      >
        <DatabaseIcon aria-hidden className="size-3 text-foreground/40" />
        {searching ? (
          <ShimmerLabel className="relative inline-block leading-none">
            {t("evidence.searching")}
          </ShimmerLabel>
        ) : (
          <span>{t("evidence.read", { count: chunks.length })}</span>
        )}
        <ChevronDownIcon
          aria-hidden
          className="size-3 opacity-60 transition-transform duration-200 group-data-panel-open/trigger:rotate-180 motion-reduce:transition-none"
        />
      </CollapsibleTrigger>
      <CollapsibleContent className={cn(collapsePanel, "outline-none")}>
        <div className="flex flex-col gap-1.5 pt-2">
          {chunks.map((chunk) => (
            <div
              key={chunk.id}
              className={cn(
                paper,
                "flex animate-in flex-col gap-1.5 rounded-2xl px-3.5 py-2.5 duration-300 fill-mode-both fade-in slide-in-from-bottom-1"
              )}
            >
              <div className="flex items-baseline gap-2">
                <span className="min-w-0 flex-1 truncate text-[13px] font-medium text-foreground/90">
                  {chunk.source}
                </span>
                <span className={cn(mono, "shrink-0 text-foreground/30")}>
                  {chunk.locator}
                </span>
              </div>
              <p className="line-clamp-2 text-xs leading-relaxed text-foreground/55">
                {chunk.text}
              </p>
            </div>
          ))}
        </div>
      </CollapsibleContent>
    </Collapsible>
  )
}
