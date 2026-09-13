from datetime import timedelta

from pharma_agent.application.conversation.ui_message import (
    EvidenceItem,
    UIMessage,
    evidence_items_from_event,
    source_document_part,
    source_document_parts_from_event,
    ui_message_of,
)
from pharma_agent.domain.conversation.models import Message, MessageRole
from pharma_agent.domain.feedback.models import Feedback, Rating
from tests.citations import (
    CURRENT_RELEASE_ID,
    OLD_RELEASE_ID,
    SNIPPET,
    SOURCE_TITLE,
    build_citation,
    chunk_id,
)
from tests.domain.factories import NOW

EXPECTED_SOURCE = {
    "type": "source-document",
    "sourceId": "00000000-0000-0000-0000-000000001001",
    "mediaType": "text/markdown",
    "title": "Paracetamol › Liều lượng và cách dùng",
    "providerMetadata": {
        "pharma": {
            "index": 1,
            "source": SOURCE_TITLE,
            "title": "Paracetamol",
            "section": "Liều lượng và cách dùng",
            "startPage": 812,
            "endPage": 813,
            "snippet": SNIPPET,
            "isCurrent": True,
        }
    },
}


def test_source_document_part_matches_the_spec_shape() -> None:
    part = source_document_part(build_citation(), is_current=True)
    assert part.model_dump(mode="json") == EXPECTED_SOURCE


def test_citations_event_items_build_current_sources() -> None:
    data = {"items": [build_citation().model_dump(mode="json")]}
    parts = source_document_parts_from_event(data)
    assert [part.model_dump(mode="json") for part in parts] == [EXPECTED_SOURCE]


def test_evidence_items_are_camel_case_with_null_pages() -> None:
    data = {
        "items": [
            {
                "index": 1,
                "source": SOURCE_TITLE,
                "title": "Paracetamol",
                "section": "Liều lượng và cách dùng",
                "start_page": None,
                "end_page": None,
                "snippet": SNIPPET,
            }
        ]
    }
    items = evidence_items_from_event(data)
    assert [item.model_dump(mode="json") for item in items] == [
        {
            "index": 1,
            "source": SOURCE_TITLE,
            "title": "Paracetamol",
            "section": "Liều lượng và cách dùng",
            "startPage": None,
            "endPage": None,
            "snippet": SNIPPET,
        }
    ]


def test_user_message_has_one_text_part_and_status_metadata() -> None:
    message = Message(
        message_id="1" * 32,
        conversation_id="c" * 32,
        role=MessageRole.USER,
        content="Paracetamol uống bao nhiêu?",
        status="completed",
        run_id="r" * 32,
        created_at=NOW,
    )
    ui = ui_message_of(message, feedback=None, current_release_ids=set())
    assert ui.model_dump(mode="json") == {
        "id": "1" * 32,
        "role": "user",
        "parts": [{"type": "text", "text": "Paracetamol uống bao nhiêu?"}],
        "metadata": {
            "status": "completed",
            "createdAt": "2026-09-11T12:00:00Z",
            "errorCode": None,
            "usage": None,
            "runId": None,
            "persisted": None,
            "feedback": None,
        },
    }


def test_assistant_message_carries_sources_feedback_and_is_current() -> None:
    message = Message(
        message_id="2" * 32,
        conversation_id="c" * 32,
        role=MessageRole.ASSISTANT,
        content="Người lớn 0,5–1 g [1], trẻ em theo cân nặng [2].",
        status="completed",
        citations=[
            build_citation(1, chunk=1),
            build_citation(2, chunk=2, release_id=OLD_RELEASE_ID),
        ],
        usage={
            "llm_calls": 5,
            "prompt_tokens": 1200,
            "completion_tokens": 80,
            "search_rounds": 1,
        },
        run_id="r" * 32,
        created_at=NOW + timedelta(microseconds=1),
    )
    feedback = Feedback.create(
        user_id="a" * 32,
        message_id="2" * 32,
        rating=Rating.DOWN,
        note="thiếu liều trẻ em",
        now=NOW,
    )

    ui = ui_message_of(
        message, feedback=feedback, current_release_ids={CURRENT_RELEASE_ID}
    )
    dumped = ui.model_dump(mode="json")

    assert [part["type"] for part in dumped["parts"]] == [
        "text",
        "source-document",
        "source-document",
    ]
    assert dumped["parts"][0] == {"type": "text", "text": message.content}
    assert [
        part["providerMetadata"]["pharma"]["isCurrent"] for part in dumped["parts"][1:]
    ] == [True, False]
    assert dumped["parts"][2]["sourceId"] == str(chunk_id(2))
    assert dumped["metadata"] == {
        "status": "completed",
        "createdAt": "2026-09-11T12:00:00.000001Z",
        "errorCode": None,
        "usage": {
            "llmCalls": 5,
            "promptTokens": 1200,
            "completionTokens": 80,
            "searchRounds": 1,
        },
        "runId": "r" * 32,
        "persisted": None,
        "feedback": {"rating": "down", "note": "thiếu liều trẻ em"},
    }
    assert UIMessage.model_validate(dumped) == ui


def test_json_schema_uses_camel_case_names() -> None:
    defs = UIMessage.model_json_schema(mode="serialization")["$defs"]
    assert set(defs["SourceDocumentUIPart"]["properties"]) == {
        "type",
        "sourceId",
        "mediaType",
        "title",
        "providerMetadata",
    }
    assert set(defs["PharmaSourceMetadata"]["properties"]) == {
        "index",
        "source",
        "title",
        "section",
        "startPage",
        "endPage",
        "snippet",
        "isCurrent",
    }
    assert set(defs["MessageMetadata"]["properties"]) == {
        "status",
        "createdAt",
        "errorCode",
        "usage",
        "runId",
        "persisted",
        "feedback",
    }
    assert defs["MessageStatus"]["enum"] == [
        "completed",
        "partial",
        "abstained",
        "blocked",
        "redirected",
        "error",
        "timeout",
    ]
    assert set(EvidenceItem.model_json_schema(mode="serialization")["properties"]) == {
        "index",
        "source",
        "title",
        "section",
        "startPage",
        "endPage",
        "snippet",
    }
