import type { ChatInit, UIMessage } from "ai"
import { z } from "zod"

import type { MessageMetadata, PharmaDataParts } from "~/api/gen/schemas"
import { ListMessagesResponse } from "~/api/gen/zod"

export const AGENT_PHASES = [
  "guarding",
  "understanding",
  "selecting_skills",
  "searching",
  "reading",
  "answering",
] as const

export type AgentPhase = (typeof AGENT_PHASES)[number]

const evidenceItemSchema = z.object({
  index: z.number().int(),
  source: z.string(),
  title: z.string(),
  section: z.string(),
  startPage: z.number().int().nullable(),
  endPage: z.number().int().nullable(),
  snippet: z.string(),
})

export const pharmaSourceMetadataSchema = evidenceItemSchema.extend({
  isCurrent: z.boolean(),
})

export type PharmaDataPartTypes = {
  [Name in keyof PharmaDataParts]: PharmaDataParts[Name]
}

export type PharmaUIMessage = UIMessage<MessageMetadata, PharmaDataPartTypes>

export type PharmaMessagePart = PharmaUIMessage["parts"][number]

export const pharmaDataPartSchemas = {
  phase: z.object({
    phase: z.enum(AGENT_PHASES),
    round: z.number().int().nullable().optional(),
  }),
  skills: z.object({
    skills: z.array(z.object({ name: z.string(), title: z.string() })),
  }),
  evidence: z.object({ items: z.array(evidenceItemSchema) }),
  conversation: z.object({ id: z.string(), title: z.string() }),
} satisfies NonNullable<ChatInit<PharmaUIMessage>["dataPartSchemas"]>

export const messageMetadataSchema = ListMessagesResponse.shape.items.element
  .shape.metadata satisfies NonNullable<
  ChatInit<PharmaUIMessage>["messageMetadataSchema"]
>
