import type {
  EvidenceData,
  MessageFeedback,
  PharmaSourceMetadata,
  PhaseData,
  SkillsData,
} from "~/api/gen/schemas"

import { textOf } from "./chat-messages"
import type { AgentPhase, PharmaUIMessage } from "./message-schema"
import { citationSourcesOf } from "./sources"

export type MessageNotice =
  | { kind: "guardrail"; reason: "blocked" | "redirected" }
  | { kind: "error"; reason: "timeout" | "failed" | "network" }
  | null

export type MessageView = {
  text: string
  hasText: boolean
  phase: PhaseData | undefined
  showStatus: boolean
  skills: SkillsData["skills"]
  evidence: EvidenceData["items"]
  sources: PharmaSourceMetadata[]
  notice: MessageNotice
  feedback: MessageFeedback | null
  canGiveFeedback: boolean
}

export const PHASE_KEYS = {
  guarding: "phase.guarding",
  understanding: "phase.understanding",
  selecting_skills: "phase.selecting_skills",
  searching: "phase.searching",
  reading: "phase.reading",
  answering: "phase.answering",
} as const satisfies Record<AgentPhase, string>

function noticeOf(message: PharmaUIMessage, failed: boolean): MessageNotice {
  const status = message.metadata?.status
  if (status === "blocked" || status === "redirected") {
    return { kind: "guardrail", reason: status }
  }
  if (status === "timeout") {
    return { kind: "error", reason: "timeout" }
  }
  if (status === "error") {
    return { kind: "error", reason: "failed" }
  }
  return failed ? { kind: "error", reason: "network" } : null
}

export function describeMessage(
  message: PharmaUIMessage,
  { streaming, failed }: { streaming: boolean; failed: boolean }
): MessageView {
  let phase: PhaseData | undefined
  let skills: SkillsData["skills"] = []
  let evidence: EvidenceData["items"] = []
  for (const part of message.parts) {
    if (part.type === "data-phase") {
      phase = part.data
    } else if (part.type === "data-skills") {
      skills = part.data.skills
    } else if (part.type === "data-evidence") {
      evidence = part.data.items
    }
  }
  const text = textOf(message)
  const notice = message.role === "assistant" ? noticeOf(message, failed) : null
  const metadata = message.metadata
  return {
    text,
    hasText: text.trim().length > 0,
    phase,
    showStatus: streaming && phase !== undefined,
    skills,
    evidence,
    sources: [...citationSourcesOf(message).values()].toSorted(
      (left, right) => left.index - right.index
    ),
    notice,
    feedback: metadata?.feedback ?? null,
    canGiveFeedback:
      message.role === "assistant" &&
      !streaming &&
      metadata !== undefined &&
      metadata.persisted !== false &&
      notice?.kind !== "error",
  }
}
