import { describe, expect, test } from "vitest"

import {
  CONTRACT_SCENARIOS,
  type ContractScenario,
  dataPartMatchesSchema,
  fixtureText,
  parseFixtureChunks,
  validChunks,
} from "../../../../tests/chat/ui-stream-fixtures"
import {
  messageMetadataSchema,
  pharmaSourceMetadataSchema,
} from "../lib/message-schema"

describe.each(CONTRACT_SCENARIOS)(
  "backend UI stream fixture %s",
  (scenario) => {
    test("parses every chunk with uiMessageChunkSchema and ends with [DONE]", async () => {
      const results = await parseFixtureChunks(scenario)
      expect(results.length).toBeGreaterThan(0)
      const unparsed = results
        .filter((result) => !result.success)
        .map((result) => result.rawValue)
      expect(unparsed).toEqual([])
      const types = validChunks(results).map((chunk) => chunk.type)
      expect(types.slice(0, 2)).toEqual(["start", "data-conversation"])
      expect(types.at(-1)).toBe("finish")
      expect(fixtureText(scenario).trimEnd().endsWith("data: [DONE]")).toBe(
        true
      )
    })

    test("matches the pharma data part, source and metadata schemas", async () => {
      const chunks = validChunks(await parseFixtureChunks(scenario))
      const mismatches = chunks.flatMap((chunk) => {
        if (chunk.type.startsWith("data-") && "data" in chunk) {
          const name = chunk.type.slice("data-".length)
          return dataPartMatchesSchema(name, chunk.data) ? [] : [chunk.type]
        }
        if (chunk.type === "source-document") {
          const pharma = chunk.providerMetadata?.["pharma"]
          return pharmaSourceMetadataSchema.safeParse(pharma).success
            ? []
            : [`${chunk.type} ${chunk.sourceId}`]
        }
        if (chunk.type === "finish") {
          return messageMetadataSchema.safeParse(chunk.messageMetadata).success
            ? []
            : [chunk.type]
        }
        return []
      })
      expect(mismatches).toEqual([])
    })
  }
)

describe("scenario specifics", () => {
  async function chunksOf(scenario: ContractScenario) {
    return validChunks(await parseFixtureChunks(scenario))
  }

  test.each([
    ["completed-with-citations", "completed", "stop", true, 2],
    ["blocked", "blocked", "stop", true, 0],
    ["timeout", "timeout", "error", true, 0],
    ["persist-failed", "completed", "stop", false, 2],
    ["no-evidence", "abstained", "stop", true, 0],
  ] as const)(
    "%s finishes with status %s, reason %s, persisted %s and %i sources",
    async (scenario, status, finishReason, persisted, sourceCount) => {
      const chunks = await chunksOf(scenario)
      const finish = chunks.find((chunk) => chunk.type === "finish")
      expect(finish?.type === "finish" ? finish.finishReason : undefined).toBe(
        finishReason
      )
      const metadata = messageMetadataSchema.parse(
        finish?.type === "finish" ? finish.messageMetadata : undefined
      )
      expect(metadata.status).toBe(status)
      expect(metadata.persisted).toBe(persisted)
      expect(
        chunks.filter((chunk) => chunk.type === "source-document")
      ).toHaveLength(sourceCount)
    }
  )
})
