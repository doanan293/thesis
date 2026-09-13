import { useMemo, useState } from "react"

import type { PharmaSourceMetadata } from "~/api/gen/schemas"
import { Sources } from "~/components/elements/sources"

export type SourcesListProps = {
  sources: readonly PharmaSourceMetadata[]
  onSelect: (index: number) => void
}

export function SourcesList({ sources, onSelect }: SourcesListProps) {
  const [open, setOpen] = useState(false)
  const cards = useMemo(
    () =>
      sources.map((source) => ({
        index: source.index,
        source: source.source,
        title: `${source.title} › ${source.section}`,
        stale: !source.isCurrent,
      })),
    [sources]
  )
  return (
    <Sources
      sources={cards}
      open={open}
      onOpenChange={setOpen}
      onSelect={onSelect}
    />
  )
}
