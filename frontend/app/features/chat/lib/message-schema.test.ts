import { describe, expect, expectTypeOf, test } from "vitest"
import type { z } from "zod"

import type {
  ConversationData,
  EvidenceData,
  MessageMetadata,
  PharmaSourceMetadata,
  PhaseData,
  SkillsData,
} from "~/api/gen/schemas"

import {
  messageMetadataSchema,
  pharmaDataPartSchemas,
  pharmaSourceMetadataSchema,
} from "./message-schema"

describe("message schemas", () => {
  test("agree with the generated types in both directions", () => {
    type Phase = z.infer<typeof pharmaDataPartSchemas.phase>
    type Skills = z.infer<typeof pharmaDataPartSchemas.skills>
    type Evidence = z.infer<typeof pharmaDataPartSchemas.evidence>
    type Conversation = z.infer<typeof pharmaDataPartSchemas.conversation>
    type Source = z.infer<typeof pharmaSourceMetadataSchema>
    expectTypeOf<Phase>().toExtend<PhaseData>()
    expectTypeOf<PhaseData>().toExtend<Phase>()
    expectTypeOf<Skills>().toExtend<SkillsData>()
    expectTypeOf<SkillsData>().toExtend<Skills>()
    expectTypeOf<Evidence>().toExtend<EvidenceData>()
    expectTypeOf<EvidenceData>().toExtend<Evidence>()
    expectTypeOf<Conversation>().toExtend<ConversationData>()
    expectTypeOf<ConversationData>().toExtend<Conversation>()
    expectTypeOf<Source>().toExtend<PharmaSourceMetadata>()
    expectTypeOf<PharmaSourceMetadata>().toExtend<Source>()
    expectTypeOf<
      z.infer<typeof messageMetadataSchema>
    >().toExtend<MessageMetadata>()
  })

  test("accept the stream data parts of spec A §3.2", () => {
    expect(
      pharmaDataPartSchemas.phase.parse({ phase: "searching", round: 2 })
    ).toEqual({ phase: "searching", round: 2 })
    expect(
      pharmaDataPartSchemas.phase.parse({ phase: "answering" }).phase
    ).toBe("answering")
    expect(
      pharmaDataPartSchemas.skills.parse({
        skills: [{ name: "drug-monograph", title: "Chuyên luận thuốc" }],
      }).skills
    ).toHaveLength(1)
    expect(
      pharmaDataPartSchemas.conversation.parse({
        id: "c1",
        title: "Liều paracetamol",
      }).title
    ).toBe("Liều paracetamol")
    expect(
      pharmaDataPartSchemas.evidence.parse({
        items: [
          {
            index: 1,
            source: "Dược thư",
            title: "Paracetamol",
            section: "Liều dùng",
            startPage: null,
            endPage: null,
            snippet: "…",
          },
        ],
      }).items[0]?.startPage
    ).toBeNull()
  })

  test("reject an unknown phase", () => {
    expect(
      pharmaDataPartSchemas.phase.safeParse({ phase: "thinking" }).success
    ).toBe(false)
  })

  test("parse finish metadata from the stream", () => {
    const metadata = messageMetadataSchema.parse({
      status: "completed",
      errorCode: null,
      usage: {
        llmCalls: 4,
        promptTokens: 900,
        completionTokens: 120,
        searchRounds: 1,
      },
      runId: "00000000000000000000000000000003",
      persisted: false,
      createdAt: "2026-09-13T08:00:00Z",
    })
    expect(metadata.persisted).toBe(false)
  })
})
