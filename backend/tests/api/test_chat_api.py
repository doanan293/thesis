from pharma_agent.application.conversation.ui_message import (
    ConversationData,
    EvidenceData,
    PhaseData,
    SkillsData,
)
from pharma_agent.domain.agent.schemas import (
    Audience,
    Intent,
    JudgeDecision,
    JudgeOutcome,
    Language,
    RephraseResult,
    SkillSelection,
)
from pharma_agent.domain.conversation.models import ConversationSummary
from pharma_agent.domain.guardrail.models import LlmGuardVerdict
from pharma_agent.domain.llm.models import LlmRole
from tests.api.harness import OWNER, build_harness, ui_chunks
from tests.contract.invariants import assert_stream_invariants
from tests.fakes import FakeLlm


def script_turn(llm: FakeLlm) -> None:
    llm.script(
        LlmRole.GUARDRAIL, LlmGuardVerdict(is_attack=False, in_scope=True, reason="ok")
    )
    llm.script(
        LlmRole.REPHRASE,
        RephraseResult(
            standalone_query="Liều paracetamol",
            audience=Audience.GENERAL_PUBLIC,
            language=Language.VI,
            intent=Intent.PHARMA_QUESTION,
        ),
    )
    llm.script(LlmRole.SKILL_SELECTOR, SkillSelection(skill_names=[]))
    llm.script(
        LlmRole.JUDGE, JudgeDecision(decision=JudgeOutcome.ANSWER, gaps=[], reason="đủ")
    )


async def test_stream_speaks_ui_message_stream_v1() -> None:
    harness = build_harness()
    script_turn(harness.llm)
    async with harness.client() as client:
        response = await client.post(
            "/api/v1/chat/stream", json={"message": "Paracetamol uống bao nhiêu?"}
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-vercel-ai-ui-message-stream"] == "v1"
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert "event:" not in response.text

    chunks = ui_chunks(response.text)
    assert_stream_invariants(chunks)
    assert {"data-phase", "data-evidence", "text-delta", "source-document"} <= {
        chunk["type"] for chunk in chunks
    }
    conversation_id = str(chunks[1]["data"]["id"])
    _, assistant = harness.repo.message_log[conversation_id]
    assert chunks[0] == {"type": "start", "messageId": assistant.message_id}
    assert [
        chunk["providerMetadata"]["pharma"]["index"]
        for chunk in chunks
        if chunk["type"] == "source-document"
    ] == [citation.index for citation in assistant.citations]
    finish = chunks[-1]
    assert finish["finishReason"] == "stop"
    assert finish["messageMetadata"]["status"] == "completed"
    assert finish["messageMetadata"]["persisted"] is True
    assert finish["messageMetadata"]["runId"] == assistant.run_id
    assert harness.repo.rows[conversation_id].user_id == OWNER.hex


async def test_stream_errors_before_streaming_are_problem_json() -> None:
    harness = build_harness()
    async with harness.client() as client:
        response = await client.post(
            "/api/v1/chat/stream",
            json={"message": "hi", "conversation_id": "f" * 32},
        )
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["code"] == "CONVERSATION_NOT_FOUND"


async def test_non_stream_chat_and_background_summary() -> None:
    harness = build_harness()
    script_turn(harness.llm)
    script_turn(harness.llm)
    harness.llm.script(
        LlmRole.SUMMARIZER, ConversationSummary(summary="Hỏi về liều paracetamol.")
    )
    async with harness.client() as client:
        first = await client.post("/api/v1/chat", json={"message": "Paracetamol?"})
        assert first.status_code == 200
        body = first.json()
        assert body["persisted"] is True and body["citations"]
        second = await client.post(
            "/api/v1/chat",
            json={"message": "Còn trẻ em?", "conversation_id": body["conversation_id"]},
        )
        assert second.status_code == 200
    stored = harness.repo.rows[body["conversation_id"]]
    assert stored.turn_count == 2 and stored.summary == "Hỏi về liều paracetamol."


async def test_chat_errors() -> None:
    harness = build_harness()
    async with harness.client() as client:
        missing = await client.post(
            "/api/v1/chat", json={"message": "hi", "conversation_id": "f" * 32}
        )
        assert (
            missing.status_code == 404
            and missing.json()["code"] == "CONVERSATION_NOT_FOUND"
        )
        invalid_id = await client.post(
            "/api/v1/chat", json={"message": "hi", "conversation_id": "nope"}
        )
        assert invalid_id.status_code == 422
        empty = await client.post("/api/v1/chat", json={"message": ""})
        assert empty.status_code == 422

    unavailable = build_harness(agent=False)
    async with unavailable.client() as client:
        response = await client.post("/api/v1/chat/stream", json={"message": "hi"})
        assert (
            response.status_code == 503
            and response.json()["code"] == "AGENT_UNAVAILABLE"
        )

    # Unauthenticated access (401) needs the user database, so it is covered by
    # tests/api/test_e2e_postgres.py instead of this database-free harness.


async def test_streamed_data_parts_match_the_documented_models() -> None:
    harness = build_harness()
    script_turn(harness.llm)
    async with harness.client() as client:
        response = await client.post(
            "/api/v1/chat/stream", json={"message": "Paracetamol?"}
        )

    models = {
        "data-phase": PhaseData,
        "data-skills": SkillsData,
        "data-evidence": EvidenceData,
        "data-conversation": ConversationData,
    }
    data_chunks = [
        chunk for chunk in ui_chunks(response.text) if chunk["type"] in models
    ]
    assert {chunk["type"] for chunk in data_chunks} >= {
        "data-phase",
        "data-evidence",
        "data-conversation",
    }
    for chunk in data_chunks:
        payload = models[chunk["type"]].model_validate(chunk["data"])
        assert set(chunk["data"]) <= set(payload.model_dump(mode="json"))
