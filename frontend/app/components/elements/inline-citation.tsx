import { PreviewCard } from "@base-ui/react/preview-card"

import { cn } from "~/lib/utils"

import { floating, mono } from "./surfaces"

export interface CitationPreview {
  source: string
  heading: string
  pages: string | null
  snippet: string
  note: string | null
}

export type CitationProps = {
  index: number
  preview: CitationPreview
  open: boolean
  onOpenChange: (open: boolean) => void
  onSelect: () => void
}

export function Citation({
  index,
  preview,
  open,
  onOpenChange,
  onSelect,
}: CitationProps) {
  return (
    <PreviewCard.Root open={open} onOpenChange={onOpenChange}>
      <PreviewCard.Trigger
        delay={0}
        render={<button type="button">{index}</button>}
        onClick={onSelect}
        className={cn(
          "mx-0.5 inline-flex h-4 min-w-4 translate-y-[-2px] cursor-pointer items-center justify-center rounded-[5px] px-1 align-middle font-mono text-[10px] font-medium tabular-nums transition-colors",
          open
            ? "bg-foreground text-background"
            : "bg-foreground/[0.06] text-foreground/45 hover:text-foreground/90"
        )}
      />
      <PreviewCard.Portal>
        <PreviewCard.Positioner side="top" sideOffset={8}>
          <PreviewCard.Popup
            className={cn(
              floating,
              "z-50 w-72 origin-(--transform-origin) rounded-2xl p-3.5 outline-none",
              "transition-[opacity,scale] duration-200 ease-[cubic-bezier(0.23,1,0.32,1)] motion-reduce:transition-none",
              "data-[starting-style]:scale-[0.97] data-[starting-style]:opacity-0",
              "data-[ending-style]:scale-[0.97] data-[ending-style]:opacity-0"
            )}
          >
            <div className="flex items-center gap-1.5">
              <span
                aria-hidden
                className="flex h-4 min-w-4 items-center justify-center rounded bg-foreground/[0.06] px-1 text-[9px] font-medium text-foreground/45 tabular-nums"
              >
                {index}
              </span>
              <span className={cn(mono, "line-clamp-1 text-foreground/40")}>
                {preview.source}
              </span>
            </div>
            <p className="mt-2 text-[13px] leading-snug font-medium">
              {preview.heading}
            </p>
            {preview.pages === null ? null : (
              <p className={cn(mono, "mt-0.5 text-foreground/40")}>
                {preview.pages}
              </p>
            )}
            <p className="mt-1 text-[13px] leading-relaxed text-foreground/50">
              {preview.snippet}
            </p>
            {preview.note === null ? null : (
              <p className="mt-1.5 text-xs text-amber-600 dark:text-amber-400">
                {preview.note}
              </p>
            )}
          </PreviewCard.Popup>
        </PreviewCard.Positioner>
      </PreviewCard.Portal>
    </PreviewCard.Root>
  )
}
