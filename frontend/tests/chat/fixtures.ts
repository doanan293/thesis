import type {
  UIMessage as ApiUIMessage,
  CitationDetail,
  ConversationPage,
  ConversationView,
  MessageMetadata,
  MessagePage,
  PharmaSourceMetadata,
} from "~/api/gen/schemas"

export const CREATED_AT = "2026-09-13T08:00:00Z"

export function pharmaSource(
  index: number,
  overrides: Partial<PharmaSourceMetadata> = {}
): PharmaSourceMetadata {
  return {
    index,
    source: "Dược thư Quốc gia Việt Nam (Xuất bản lần thứ 2)",
    title: "Paracetamol",
    section: "Liều lượng và cách dùng",
    startPage: 812,
    endPage: 813,
    snippet: "Người lớn và trẻ em trên 12 tuổi uống 0,5–1 g mỗi 4–6 giờ.",
    isCurrent: true,
    ...overrides,
  }
}

export function apiUserMessage(id: string, text: string): ApiUIMessage {
  return {
    id,
    role: "user",
    parts: [{ type: "text", text }],
    metadata: { status: "completed", createdAt: CREATED_AT },
  }
}

export function apiAssistantMessage(
  id: string,
  text: string,
  sources: readonly PharmaSourceMetadata[] = [],
  metadata: Partial<MessageMetadata> = {}
): ApiUIMessage {
  return {
    id,
    role: "assistant",
    parts: [
      { type: "text", text },
      ...sources.map((source) => ({
        type: "source-document" as const,
        sourceId: `chunk-${source.index}`,
        mediaType: "text/markdown" as const,
        title: `${source.title} › ${source.section}`,
        providerMetadata: { pharma: source },
      })),
    ],
    metadata: {
      status: "completed",
      createdAt: CREATED_AT,
      feedback: null,
      ...metadata,
    },
  }
}

export function conversationTurns(
  count: number,
  prefix: string
): ApiUIMessage[] {
  return Array.from({ length: count }, (_unused, position) => {
    const turn = position + 1
    return [
      apiUserMessage(`${prefix}-u${turn}`, `Câu hỏi ${prefix} ${turn}`),
      apiAssistantMessage(
        `${prefix}-a${turn}`,
        `Trả lời ${prefix} ${turn}\n\n${"Nội dung chi tiết. ".repeat(12)}`
      ),
    ]
  }).flat()
}

export function messagePage(
  items: ApiUIMessage[],
  nextCursor: string | null = null
): MessagePage {
  return { items, next_cursor: nextCursor }
}

export function conversationView(
  id: string,
  title: string,
  overrides: Partial<ConversationView> = {}
): ConversationView {
  return {
    id,
    title,
    turn_count: 1,
    created_at: CREATED_AT,
    updated_at: CREATED_AT,
    ...overrides,
  }
}

export function conversationPage(
  items: ConversationView[],
  nextCursor: string | null = null
): ConversationPage {
  return { items, next_cursor: nextCursor }
}

export function citationDetail(
  index: number,
  overrides: Partial<CitationDetail> = {}
): CitationDetail {
  return {
    index,
    source: "Dược thư Quốc gia Việt Nam (Xuất bản lần thứ 2)",
    document_title: "Paracetamol",
    section: "Liều lượng và cách dùng",
    start_page: 811,
    end_page: 812,
    strategy: "full_section",
    is_current: true,
    chunks: [
      {
        id: "7b0f4f4e-2c1f-4f0e-9a5e-000000000000",
        text: "**Liều thường dùng**",
        matched: false,
        start_page: 811,
        end_page: 811,
      },
      {
        id: "7b0f4f4e-2c1f-4f0e-9a5e-000000000001",
        text: "Người lớn và trẻ em trên 12 tuổi uống 0,5–1 g mỗi 4–6 giờ.",
        matched: true,
        start_page: 812,
        end_page: 812,
      },
    ],
    ...overrides,
  }
}
