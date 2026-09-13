import { useMemo, useState } from "react"

import type { EvidenceData } from "~/api/gen/schemas"
import { RetrievalChunks } from "~/components/elements/retrieval-chunks"
import { usePageText } from "~/features/citations/lib/citation-format"

export type EvidencePanelProps = {
  items: EvidenceData["items"]
  searching: boolean
}

export function EvidencePanel({ items, searching }: EvidencePanelProps) {
  const pageText = usePageText()
  const [openOverride, setOpenOverride] = useState<boolean | null>(null)
  const chunks = useMemo(
    () =>
      items.map((item) => ({
        id: String(item.index),
        source: `${item.title} › ${item.section}`,
        locator: pageText(item.startPage, item.endPage) ?? "",
        text: item.snippet,
      })),
    [items, pageText]
  )
  return (
    <RetrievalChunks
      chunks={chunks}
      searching={searching}
      open={openOverride ?? searching}
      onOpenChange={setOpenOverride}
    />
  )
}
