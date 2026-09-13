"""AI SDK UI Message Stream v1 over SSE (spec A §3.2).

`UIMessageStreamEncoder` is a pure mapping from chat `ProgressEvent`s to chunk dicts; the domain
event contract is unchanged. SSE framing is sse-starlette's: one `data:` line per chunk, then
`data: [DONE]`.
"""

from collections.abc import AsyncIterable, AsyncIterator
from datetime import datetime
from typing import Any, assert_never

from sse_starlette import JSONServerSentEvent, ServerSentEvent

from pharma_agent.application.conversation.ui_message import (
    MessageMetadata,
    MessageStatus,
    MessageUsage,
    evidence_items_from_event,
    source_document_parts_from_event,
)
from pharma_agent.application.progress import EventType, ProgressEvent

UI_MESSAGE_STREAM_HEADERS = {
    "x-vercel-ai-ui-message-stream": "v1",
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}
DONE_SENTINEL = "[DONE]"
PHASE_PART_ID = "phase"
EVIDENCE_PART_ID = "evidence"
TEXT_PART_ID = "text"
_ERROR_STATUSES = frozenset({MessageStatus.ERROR, MessageStatus.TIMEOUT})

Chunk = dict[str, Any]


class UIMessageStreamEncoder:
    """One instance per turn; the only state is whether the text part is open."""

    def __init__(self) -> None:
        self._text_open = False

    def encode(self, event: ProgressEvent) -> list[Chunk]:
        data = event.data
        match event.type:
            case EventType.CONVERSATION:
                return [
                    {"type": "start", "messageId": data["message_id"]},
                    {
                        "type": "data-conversation",
                        "transient": True,
                        "data": {"id": data["conversation_id"], "title": data["title"]},
                    },
                ]
            case EventType.PHASE:
                phase: dict[str, Any] = {"phase": data["phase"]}
                if "round" in data:
                    phase["round"] = data["round"]
                return [{"type": "data-phase", "id": PHASE_PART_ID, "data": phase}]
            case EventType.SKILLS_SELECTED:
                skills = [
                    {"name": skill["name"], "title": skill["title"]}
                    for skill in data["skills"]
                ]
                return [{"type": "data-skills", "data": {"skills": skills}}]
            case EventType.EVIDENCE:
                items = [
                    item.model_dump(mode="json")
                    for item in evidence_items_from_event(data)
                ]
                return [
                    {
                        "type": "data-evidence",
                        "id": EVIDENCE_PART_ID,
                        "data": {"items": items},
                    }
                ]
            case EventType.TOKEN:
                return self._text(str(data["text"]))
            case EventType.CITATIONS:
                chunks = self._close_text()
                chunks.extend(
                    part.model_dump(mode="json")
                    for part in source_document_parts_from_event(data)
                )
                return chunks
            case EventType.DONE:
                return [*self._close_text(), _finish(data)]
            case _:
                assert_never(event.type)

    def _text(self, text: str) -> list[Chunk]:
        if not text:
            return []
        chunks: list[Chunk] = []
        if not self._text_open:
            self._text_open = True
            chunks.append({"type": "text-start", "id": TEXT_PART_ID})
        chunks.append({"type": "text-delta", "id": TEXT_PART_ID, "delta": text})
        return chunks

    def _close_text(self) -> list[Chunk]:
        if not self._text_open:
            return []
        self._text_open = False
        return [{"type": "text-end", "id": TEXT_PART_ID}]


def _finish(data: dict[str, Any]) -> Chunk:
    status = MessageStatus(data["status"])
    metadata = MessageMetadata(
        status=status,
        created_at=datetime.fromisoformat(data["created_at"]),
        error_code=data["error_code"],
        usage=MessageUsage.model_validate(data["usage"]),
        run_id=data["run_id"],
        persisted=data["persisted"],
    )
    return {
        "type": "finish",
        "finishReason": "error" if status in _ERROR_STATUSES else "stop",
        "messageMetadata": metadata.model_dump(mode="json", exclude={"feedback"}),
    }


async def ui_message_chunks(
    events: AsyncIterable[ProgressEvent],
) -> AsyncIterator[Chunk]:
    encoder = UIMessageStreamEncoder()
    async for event in events:
        for chunk in encoder.encode(event):
            yield chunk


async def ui_message_stream(
    events: AsyncIterable[ProgressEvent],
) -> AsyncIterator[ServerSentEvent]:
    async for chunk in ui_message_chunks(events):
        yield JSONServerSentEvent(data=chunk)
    yield ServerSentEvent(data=DONE_SENTINEL)
