import type { SourceDocumentUIPart } from "ai"

import type { PharmaSourceMetadata } from "~/api/gen/schemas"

import {
  type PharmaUIMessage,
  pharmaSourceMetadataSchema,
} from "./message-schema"

export function readPharmaSource(
  part: SourceDocumentUIPart
): PharmaSourceMetadata | undefined {
  const parsed = pharmaSourceMetadataSchema.safeParse(
    part.providerMetadata?.["pharma"]
  )
  return parsed.success ? parsed.data : undefined
}

export function citationSourcesOf(
  message: PharmaUIMessage
): Map<number, PharmaSourceMetadata> {
  const sources = new Map<number, PharmaSourceMetadata>()
  for (const part of message.parts) {
    if (part.type === "source-document") {
      const source = readPharmaSource(part)
      if (source !== undefined) {
        sources.set(source.index, source)
      }
    }
  }
  return sources
}
