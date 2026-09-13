import { useState } from "react"
import { useTranslation } from "react-i18next"

import { Citation } from "~/components/elements/inline-citation"
import { parseCiteIndex } from "~/features/chat/lib/cite-markers"

import { useMessageCitations } from "./citation-context"
import { usePageText } from "./lib/citation-format"

export function CitationMarker(props: Record<string, unknown>) {
  const citations = useMessageCitations()
  const { t } = useTranslation("chat")
  const pageText = usePageText()
  const [open, setOpen] = useState(false)
  const index = parseCiteIndex(props["index"])
  if (index === null) {
    return null
  }
  const source = citations?.sources.get(index)
  if (citations === null || source === undefined) {
    return <span className="text-muted-foreground">{`[${index}]`}</span>
  }
  return (
    <Citation
      index={index}
      open={open}
      onOpenChange={setOpen}
      onSelect={() => {
        setOpen(false)
        citations.openCitation(index)
      }}
      preview={{
        source: source.source,
        heading: t("citation.excerpt", {
          title: source.title,
          section: source.section,
        }),
        pages: pageText(source.startPage, source.endPage),
        snippet: source.snippet,
        note: source.isCurrent ? null : t("sources.stale"),
      }}
    />
  )
}
