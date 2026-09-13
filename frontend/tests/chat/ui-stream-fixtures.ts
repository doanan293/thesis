import {
  parseJsonEventStream,
  readUIMessageStream,
  type UIMessageChunk,
  uiMessageChunkSchema,
} from "ai"

import {
  pharmaDataPartSchemas,
  type PharmaUIMessage,
} from "~/features/chat/lib/message-schema"

const fixtureFiles = import.meta.glob<string>(
  "../../../backend/tests/contract/fixtures/ui-stream/*.sse",
  { query: "?raw", import: "default", eager: true }
)

export const CONTRACT_SCENARIOS = [
  "completed-with-citations",
  "blocked",
  "timeout",
  "persist-failed",
  "no-evidence",
] as const

export type ContractScenario = (typeof CONTRACT_SCENARIOS)[number]

type StreamValue<S> = S extends ReadableStream<infer V> ? V : never

export type ChunkParseResult = StreamValue<
  ReturnType<typeof parseJsonEventStream<UIMessageChunk>>
>

export function fixtureText(scenario: ContractScenario): string {
  const entry = Object.entries(fixtureFiles).find(([path]) =>
    path.endsWith(`/${scenario}.sse`)
  )
  if (entry === undefined) {
    throw new Error(
      `Missing backend contract fixture ${scenario}.sse; run the backend contract tests first`
    )
  }
  return entry[1]
}

function textStream(text: string): ReadableStream<Uint8Array> {
  const bytes = new TextEncoder().encode(text)
  return new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(bytes)
      controller.close()
    },
  })
}

async function readAll<T>(stream: ReadableStream<T>): Promise<T[]> {
  const reader = stream.getReader()
  const values: T[] = []
  let result = await reader.read()
  while (!result.done) {
    values.push(result.value)
    result = await reader.read()
  }
  return values
}

export function parseFixtureChunks(
  scenario: ContractScenario
): Promise<ChunkParseResult[]> {
  return readAll(
    parseJsonEventStream({
      stream: textStream(fixtureText(scenario)),
      schema: uiMessageChunkSchema,
    })
  )
}

export function validChunks(
  results: readonly ChunkParseResult[]
): UIMessageChunk[] {
  return results.flatMap((result) => (result.success ? [result.value] : []))
}

export function dataPartMatchesSchema(name: string, data: unknown): boolean {
  switch (name) {
    case "phase":
      return pharmaDataPartSchemas.phase.safeParse(data).success
    case "skills":
      return pharmaDataPartSchemas.skills.safeParse(data).success
    case "evidence":
      return pharmaDataPartSchemas.evidence.safeParse(data).success
    case "conversation":
      return pharmaDataPartSchemas.conversation.safeParse(data).success
    default:
      return false
  }
}

export async function fixtureMessage(
  scenario: ContractScenario
): Promise<PharmaUIMessage> {
  const chunks = validChunks(await parseFixtureChunks(scenario))
  const stream = new ReadableStream<UIMessageChunk>({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(chunk)
      }
      controller.close()
    },
  })
  let last: PharmaUIMessage | undefined
  for await (const message of readUIMessageStream<PharmaUIMessage>({
    stream,
  })) {
    last = message
  }
  if (last === undefined) {
    throw new Error(`Fixture ${scenario}.sse produced no message`)
  }
  return last
}
