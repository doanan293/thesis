import pytest

from pharma_agent.application.chat.graph import build_chat_graph
from pharma_agent.application.chat.runner import ChatTurnRunner
from pharma_agent.application.chat.service import ChatService, MemoryPolicy
from pharma_agent.application.errors import ConversationNotFound
from pharma_agent.application.progress import EventType, ProgressEvent
from pharma_agent.domain.agent.budget import BudgetLimits
from pharma_agent.domain.agent.schemas import (
    Audience,
    Intent,
    JudgeDecision,
    JudgeOutcome,
    Language,
    RephraseResult,
    SkillSelection,
)
from pharma_agent.domain.guardrail.models import LlmGuardVerdict
from pharma_agent.domain.llm.models import LlmRole
from pharma_agent.domain.shared.clock import FixedClock
from tests.domain.factories import make_hit
from tests.fakes import NOW, FakeLlm, FakeRetriever, build_deps
from tests.memory_repository import InMemoryConversationRepository

OWNER = "a" * 32
STRANGER = "b" * 32


def scripted_turn(llm: FakeLlm, standalone: str) -> None:
    llm.script(
        LlmRole.GUARDRAIL, LlmGuardVerdict(is_attack=False, in_scope=True, reason="ok")
    )
    llm.script(
        LlmRole.REPHRASE,
        RephraseResult(
            standalone_query=standalone,
            audience=Audience.GENERAL_PUBLIC,
            language=Language.VI,
            intent=Intent.PHARMA_QUESTION,
        ),
    )
    llm.script(LlmRole.SKILL_SELECTOR, SkillSelection(skill_ids=[]))
    llm.script(
        LlmRole.JUDGE, JudgeDecision(decision=JudgeOutcome.ANSWER, gaps=[], reason="đủ")
    )


def service_with(
    llm: FakeLlm, repo: InMemoryConversationRepository, rounds: int = 2
) -> ChatService:
    retriever = FakeRetriever(*[[make_hit(f"c{i}", fusion=0.9)] for i in range(rounds)])
    runner = ChatTurnRunner(
        build_chat_graph(), build_deps(llm, retriever), BudgetLimits()
    )
    return ChatService(
        runner, repo, FixedClock(NOW), MemoryPolicy(context_turns=4, context_chars=4000)
    )


async def collect(events) -> list[ProgressEvent]:
    return [event async for event in events]


async def test_new_conversation_turn_is_persisted_with_audit() -> None:
    llm, repo = FakeLlm(), InMemoryConversationRepository()
    scripted_turn(llm, "Liều paracetamol cho người lớn")
    service = service_with(llm, repo)

    session = await service.open_turn(
        user_id=OWNER, message="Paracetamol uống bao nhiêu?", conversation_id=None
    )
    events = await collect(session.events())

    first = events[0]
    assert first.type is EventType.CONVERSATION
    assert first.data == {
        "conversation_id": session.conversation_id,
        "title": "Paracetamol uống bao nhiêu?",
        "created": True,
    }
    done = next(e for e in events if e.type is EventType.DONE)
    assert done.data["conversation_id"] == session.conversation_id
    assert done.data["status"] == "completed" and done.data["message_id"]
    assert events[-1] is done

    stored = repo.rows[session.conversation_id]
    assert stored.turn_count == 1 and stored.user_id == OWNER
    user_msg, assistant_msg = repo.message_log[session.conversation_id]
    assert user_msg.content == "Paracetamol uống bao nhiêu?"
    assert (
        assistant_msg.message_id == done.data["message_id"] and assistant_msg.citations
    )
    assert "answering" in assistant_msg.phases
    assert (
        repo.audit[assistant_msg.message_id][0].query_text
        == "Liều paracetamol cho người lớn"
    )
    assert session.result is not None and session.result.persisted is True


async def test_follow_up_turn_sends_previous_turn_to_rephrase() -> None:
    llm, repo = FakeLlm(), InMemoryConversationRepository()
    scripted_turn(llm, "Liều paracetamol cho người lớn")
    scripted_turn(llm, "Paracetamol có dùng cho trẻ em không")
    service = service_with(llm, repo)

    first = await service.ask(
        user_id=OWNER, message="Paracetamol uống bao nhiêu?", conversation_id=None
    )
    second_session = await service.open_turn(
        user_id=OWNER,
        message="Còn trẻ em thì sao?",
        conversation_id=first.conversation_id,
    )
    events = await collect(second_session.events())

    assert events[0].data["created"] is False
    rephrase_prompt = llm.calls_for(LlmRole.REPHRASE)[1][1].content
    assert (
        "Paracetamol uống bao nhiêu?" in rephrase_prompt
        and "Còn trẻ em thì sao?" in rephrase_prompt
    )
    assert repo.rows[first.conversation_id].turn_count == 2


async def test_unknown_or_foreign_conversation_is_not_found() -> None:
    llm, repo = FakeLlm(), InMemoryConversationRepository()
    scripted_turn(llm, "x")
    service = service_with(llm, repo)
    owned = await service.ask(user_id=OWNER, message="hi", conversation_id=None)

    with pytest.raises(ConversationNotFound):
        await service.open_turn(user_id=OWNER, message="hi", conversation_id="f" * 32)
    with pytest.raises(ConversationNotFound):
        await service.open_turn(
            user_id=STRANGER, message="hi", conversation_id=owned.conversation_id
        )


async def test_persist_failure_reports_error_after_done() -> None:
    llm, repo = FakeLlm(), InMemoryConversationRepository()
    scripted_turn(llm, "Liều paracetamol")
    service = service_with(llm, repo)
    session = await service.open_turn(
        user_id=OWNER, message="Paracetamol?", conversation_id=None
    )
    repo.fail_append = True

    events = await collect(session.events())

    assert [e.type for e in events[-2:]] == [EventType.DONE, EventType.ERROR]
    assert events[-2].data["message_id"] is None
    assert events[-1].data["code"] == "PERSIST_FAILED"
    assert session.result is not None and session.result.persisted is False
    assert session.result.content  # the answer is still delivered
