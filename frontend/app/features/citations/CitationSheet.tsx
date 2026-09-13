import { useQuery } from "@tanstack/react-query"
import { useState } from "react"
import { useTranslation } from "react-i18next"
import { Streamdown } from "streamdown"

import { getGetMessageCitationQueryOptions } from "~/api/gen/endpoints"
import type { CitationDetail } from "~/api/gen/schemas"
import { Badge } from "~/components/ui/badge"
import {
  Drawer,
  DrawerContent,
  DrawerHeader,
  DrawerTitle,
} from "~/components/ui/drawer"
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "~/components/ui/sheet"
import { Spinner } from "~/components/ui/spinner"
import { streamdownPlugins } from "~/features/chat/lib/streamdown-config"
import { useIsMobile } from "~/hooks/use-mobile"
import { apiErrorMessage } from "~/i18n/error-message"
import { cn } from "~/lib/utils"

import { usePageText } from "./lib/citation-format"

export type CitationTarget = {
  messageId: string
  index: number
}

export type CitationSheetProps = {
  target: CitationTarget | null
  onClose: () => void
}

const STRATEGY_KEYS = {
  full_section: "strategy.full_section",
  chunk_window: "strategy.chunk_window",
  search_only: "strategy.search_only",
} as const satisfies Record<CitationDetail["strategy"], string>

// Runs as a callback ref when the matched chunk mounts, which is when the citation data renders.
function scrollIntoCenter(node: HTMLElement | null) {
  node?.scrollIntoView({ block: "center" })
}

export function CitationDetailView({ messageId, index }: CitationTarget) {
  const { t } = useTranslation("citations")
  const { t: tErrors } = useTranslation("errors")
  const pageText = usePageText()
  const query = useQuery(getGetMessageCitationQueryOptions(messageId, index))

  if (query.isPending) {
    return (
      <output className="flex items-center gap-2 text-sm text-muted-foreground">
        <Spinner />
        {t("sheet.loading")}
      </output>
    )
  }

  if (query.isError) {
    return (
      <div role="alert" className="flex flex-col gap-1 text-sm">
        <p className="font-medium text-destructive">{t("sheet.loadFailed")}</p>
        <p className="text-muted-foreground">
          {apiErrorMessage(query.error, tErrors)}
        </p>
      </div>
    )
  }

  const detail = query.data
  const pages = pageText(detail.start_page, detail.end_page)
  return (
    <article className="flex flex-col gap-4">
      <header className="flex flex-col gap-1">
        <p className="text-xs text-muted-foreground">{detail.source}</p>
        <p className="text-base font-medium">
          {`${detail.document_title} › ${detail.section}`}
        </p>
        <p className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
          {pages === null ? null : <span>{pages}</span>}
          <Badge variant="outline">{t(STRATEGY_KEYS[detail.strategy])}</Badge>
        </p>
        {detail.is_current ? null : (
          <p className="text-xs text-amber-600 dark:text-amber-400">
            {t("stale")}
          </p>
        )}
      </header>
      {detail.chunks.map((chunk) => (
        <section
          key={chunk.id}
          ref={chunk.matched ? scrollIntoCenter : undefined}
          data-matched={chunk.matched}
          aria-label={chunk.matched ? t("sheet.matched") : undefined}
          className={cn(
            "rounded-xl px-3 py-2",
            chunk.matched && "bg-amber-500/10 ring-1 ring-amber-500/30"
          )}
        >
          <Streamdown
            mode="static"
            plugins={streamdownPlugins}
            className="text-sm leading-relaxed"
          >
            {chunk.text}
          </Streamdown>
        </section>
      ))}
    </article>
  )
}

export function CitationSheet({ target, onClose }: CitationSheetProps) {
  const { t } = useTranslation("citations")
  const isMobile = useIsMobile()
  const [shown, setShown] = useState(target)
  if (target !== null && target !== shown) {
    setShown(target)
  }

  const open = target !== null
  const title = shown === null ? "" : t("sheet.title", { index: shown.index })
  const body =
    shown === null ? null : (
      <CitationDetailView messageId={shown.messageId} index={shown.index} />
    )
  const handleOpenChange = (next: boolean) => {
    if (!next) {
      onClose()
    }
  }

  if (isMobile) {
    return (
      <Drawer open={open} onOpenChange={handleOpenChange}>
        <DrawerContent className="max-h-[85dvh]">
          <DrawerHeader>
            <DrawerTitle>{title}</DrawerTitle>
          </DrawerHeader>
          <div className="overflow-y-auto px-4 pb-6">{body}</div>
        </DrawerContent>
      </Drawer>
    )
  }

  return (
    <Sheet open={open} onOpenChange={handleOpenChange}>
      <SheetContent
        side="right"
        className="w-full data-[side=right]:sm:max-w-xl"
      >
        <SheetHeader>
          <SheetTitle>{title}</SheetTitle>
        </SheetHeader>
        <div className="min-h-0 flex-1 overflow-y-auto px-4 pb-6">{body}</div>
      </SheetContent>
    </Sheet>
  )
}
