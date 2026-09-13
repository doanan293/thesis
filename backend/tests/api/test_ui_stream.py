import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from pharma_agent.api.ui_stream import UIMessageStreamEncoder, ui_message_stream
from pharma_agent.application.conversation.ui_message import source_document_part
from pharma_agent.application.progress import EventType, Phase, ProgressEvent
from tests.citations import SNIPPET, SOURCE_TITLE, build_citation
from tests.contract.invariants import assert_stream_invariants

MESSAGE_ID = "1" * 32
CONVERSATION_ID = "2" * 32
RUN_ID = "3" * 32
CREATED_AT = "2026-09-13T08:00:00+00:00"
USAGE = {
    "llm_calls": 5,
    "prompt_tokens": 1200,
    "completion_tokens": 70,
    "search_rounds": 1,
}
EVIDENCE_ITEM = {
    "index": 1,
    "source": SOURCE_TITLE,
    "title": "Paracetamol",
    "section": "Liều lượng và cách dùng",
    "start_page": 812,
    "end_page": None,
    "snippet": SNIPPET,
}


def conversation_event() -> ProgressEvent:
    return ProgressEvent(
        type=EventType.CONVERSATION,
        data={
            "conversation_id": CONVERSATION_ID,
            "title": "Paracetamol?",
            "created": True,
            "message_id": MESSAGE_ID,
        },
    )


def citations_event() -> ProgressEvent:
    return ProgressEvent(
        type=EventType.CITATIONS,
        data={"items": [build_citation().model_dump(mode="json")]},
    )


def done_event(
    status: str, *, error_code: str | None = None, persisted: bool = True
) -> ProgressEvent:
    return ProgressEvent(
        type=EventType.DONE,
        data={
            "run_id": RUN_ID,
            "conversation_id": CONVERSATION_ID,
            "status": status,
            "error_code": error_code,
            "usage": USAGE,
            "message_id": MESSAGE_ID if persisted else None,
            "persisted": persisted,
            "created_at": CREATED_AT,
        },
    )


def encode_all(events: list[ProgressEvent]) -> list[dict[str, Any]]:
    encoder = UIMessageStreamEncoder()
    return [chunk for event in events for chunk in encoder.encode(event)]


def test_completed_turn_maps_every_event_in_spec_order() -> None:
    events = [
        conversation_event(),
        ProgressEvent.phase(Phase.GUARDING),
        ProgressEvent(
            type=EventType.SKILLS_SELECTED,
            data={
                "skills": [
                    {"name": "drug-monograph", "title": "Tra cứu chuyên luận thuốc"}
                ]
            },
        ),
        ProgressEvent.phase(Phase.SEARCHING, round=1),
        ProgressEvent(type=EventType.EVIDENCE, data={"items": [EVIDENCE_ITEM]}),
        ProgressEvent.token("Người lớn 0,5–1 g "),
        ProgressEvent.token("mỗi 4–6 giờ [1]."),
        citations_event(),
        done_event("completed"),
    ]

    chunks = encode_all(events)

    assert chunks == [
        {"type": "start", "messageId": MESSAGE_ID},
        {
            "type": "data-conversation",
            "transient": True,
            "data": {"id": CONVERSATION_ID, "title": "Paracetamol?"},
        },
        {"type": "data-phase", "id": "phase", "data": {"phase": "guarding"}},
        {
            "type": "data-skills",
            "data": {
                "skills": [
                    {"name": "drug-monograph", "title": "Tra cứu chuyên luận thuốc"}
                ]
            },
        },
        {
            "type": "data-phase",
            "id": "phase",
            "data": {"phase": "searching", "round": 1},
        },
        {
            "type": "data-evidence",
            "id": "evidence",
            "data": {
                "items": [
                    {
                        "index": 1,
                        "source": SOURCE_TITLE,
                        "title": "Paracetamol",
                        "section": "Liều lượng và cách dùng",
                        "startPage": 812,
                        "endPage": None,
                        "snippet": SNIPPET,
                    }
                ]
            },
        },
        {"type": "text-start", "id": "text"},
        {"type": "text-delta", "id": "text", "delta": "Người lớn 0,5–1 g "},
        {"type": "text-delta", "id": "text", "delta": "mỗi 4–6 giờ [1]."},
        {"type": "text-end", "id": "text"},
        source_document_part(build_citation(), is_current=True).model_dump(mode="json"),
        {
            "type": "finish",
            "finishReason": "stop",
            "messageMetadata": {
                "status": "completed",
                "createdAt": "2026-09-13T08:00:00Z",
                "errorCode": None,
                "usage": {
                    "llmCalls": 5,
                    "promptTokens": 1200,
                    "completionTokens": 70,
                    "searchRounds": 1,
                },
                "runId": RUN_ID,
                "persisted": True,
            },
        },
    ]
    assert_stream_invariants(chunks)


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        ("completed", "stop"),
        ("partial", "stop"),
        ("abstained", "stop"),
        ("blocked", "stop"),
        ("redirected", "stop"),
        ("error", "error"),
        ("timeout", "error"),
    ],
)
def test_finish_reason_follows_status(status: str, reason: str) -> None:
    chunks = encode_all([conversation_event(), done_event(status)])
    assert chunks[-1]["finishReason"] == reason
    assert chunks[-1]["messageMetadata"]["status"] == status
    assert_stream_invariants(chunks)


def test_text_opens_on_first_token_and_closes_before_finish_without_citations() -> None:
    chunks = encode_all(
        [
            conversation_event(),
            ProgressEvent.phase(Phase.ANSWERING),
            ProgressEvent.token(""),
            ProgressEvent.token("Hệ thống xử lý quá lâu, bạn thử lại nhé."),
            done_event("timeout", error_code="DEADLINE_EXCEEDED"),
        ]
    )
    assert [chunk["type"] for chunk in chunks] == [
        "start",
        "data-conversation",
        "data-phase",
        "text-start",
        "text-delta",
        "text-end",
        "finish",
    ]
    assert chunks[-1]["finishReason"] == "error"
    assert chunks[-1]["messageMetadata"]["errorCode"] == "DEADLINE_EXCEEDED"
    assert_stream_invariants(chunks)


def test_no_token_means_no_text_part() -> None:
    chunks = encode_all(
        [
            conversation_event(),
            ProgressEvent(type=EventType.CITATIONS, data={"items": []}),
            done_event("abstained"),
        ]
    )
    assert [chunk["type"] for chunk in chunks] == [
        "start",
        "data-conversation",
        "finish",
    ]


def test_unpersisted_turn_is_reported_in_finish_metadata_only() -> None:
    chunks = encode_all(
        [
            conversation_event(),
            ProgressEvent.token("Người lớn 500 mg."),
            done_event("completed", persisted=False),
        ]
    )
    assert chunks[-1]["messageMetadata"]["persisted"] is False
    assert all(chunk["type"] != "error" for chunk in chunks)
    assert_stream_invariants(chunks)


def test_invariants_reject_split_markers_and_orphan_sources() -> None:
    split = encode_all(
        [
            conversation_event(),
            ProgressEvent.token("Liều [1"),
            ProgressEvent.token("]."),
            citations_event(),
            done_event("completed"),
        ]
    )
    with pytest.raises(AssertionError, match="marker split"):
        assert_stream_invariants(split)

    orphan = encode_all(
        [
            conversation_event(),
            ProgressEvent.token("Liều 500 mg."),
            citations_event(),
            done_event("completed"),
        ]
    )
    with pytest.raises(AssertionError, match="markers"):
        assert_stream_invariants(orphan)


async def test_sse_frames_are_data_only_and_end_with_done() -> None:
    async def events() -> AsyncIterator[ProgressEvent]:
        yield conversation_event()
        yield ProgressEvent.token("Liều [1]")
        yield done_event("completed")

    body = b"".join([frame.encode() async for frame in ui_message_stream(events())])
    blocks = [block for block in body.decode().split("\r\n\r\n") if block]

    assert all(block.startswith("data: ") and "\r\n" not in block for block in blocks)
    assert blocks[-1] == "data: [DONE]"
    assert json.loads(blocks[0].removeprefix("data: ")) == {
        "type": "start",
        "messageId": MESSAGE_ID,
    }
    assert 'data: {"type":"text-delta","id":"text","delta":"Liều [1]"}' in blocks
