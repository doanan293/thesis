import re

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
)
from pharma_agent.domain.conversation.models import Conversation
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


async def test_open_turn_pre_generates_distinct_message_ids() -> None:
    llm, repo = FakeLlm(), InMemoryConversationRepository()
    service = service_with(llm, repo)

    session = await service.open_turn(user_id=OWNER, message="hi", conversation_id=None)

    assert re.fullmatch(r"[0-9a-f]{32}", session.user_message_id)
    assert re.fullmatch(r"[0-9a-f]{32}", session.assistant_message_id)
    assert session.user_message_id != session.assistant_message_id


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
        "message_id": session.assistant_message_id,
    }
    done = events[-1]
    assert done.type is EventType.DONE
    assert done.data["conversation_id"] == session.conversation_id
    assert done.data["status"] == "completed"
    assert done.data["message_id"] == session.assistant_message_id
    assert done.data["persisted"] is True

    stored = repo.rows[session.conversation_id]
    assert stored.turn_count == 1 and stored.user_id == OWNER
    user_msg, assistant_msg = repo.message_log[session.conversation_id]
    assert user_msg.message_id == session.user_message_id
    assert user_msg.content == "Paracetamol uống bao nhiêu?"
    assert assistant_msg.message_id == session.assistant_message_id
    assert assistant_msg.citations
    assert done.data["created_at"] == assistant_msg.created_at.isoformat()
    assert "answering" in assistant_msg.phases
    assert (
        repo.audit[assistant_msg.message_id][0].query_text
        == "Liều paracetamol cho người lớn"
    )
    assert session.result is not None and session.result.persisted is True
    assert session.result.created_at == assistant_msg.created_at


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


async def test_persist_failure_is_reported_on_done() -> None:
    llm, repo = FakeLlm(), InMemoryConversationRepository()
    scripted_turn(llm, "Liều paracetamol")
    service = service_with(llm, repo)
    session = await service.open_turn(
        user_id=OWNER, message="Paracetamol?", conversation_id=None
    )
    repo.fail_append = True

    events = await collect(session.events())

    done = events[-1]
    assert done.type is EventType.DONE
    assert done.data["persisted"] is False and done.data["message_id"] is None
    assert [event.type for event in events].count(EventType.DONE) == 1
    assert session.result is not None and session.result.persisted is False
    assert session.result.content  # the answer is still delivered


async def test_first_turn_names_a_pre_created_conversation() -> None:
    llm, repo = FakeLlm(), InMemoryConversationRepository()
    scripted_turn(llm, "Liều paracetamol cho người lớn")
    scripted_turn(llm, "Paracetamol có dùng cho trẻ em không")
    service = service_with(llm, repo)
    empty = Conversation.create_empty(user_id=OWNER, now=NOW)
    await repo.create(empty)

    session = await service.open_turn(
        user_id=OWNER,
        message="Paracetamol uống bao nhiêu?",
        conversation_id=empty.conversation_id,
    )
    events = await collect(session.events())

    assert events[0].data == {
        "conversation_id": empty.conversation_id,
        "title": "Paracetamol uống bao nhiêu?",
        "created": False,
        "message_id": session.assistant_message_id,
    }
    stored = repo.rows[empty.conversation_id]
    assert stored.title == "Paracetamol uống bao nhiêu?" and stored.turn_count == 1

    await service.ask(
        user_id=OWNER,
        message="Còn trẻ em thì sao?",
        conversation_id=empty.conversation_id,
    )
    assert repo.rows[empty.conversation_id].title == "Paracetamol uống bao nhiêu?"


async def test_renamed_empty_conversation_keeps_its_title() -> None:
    llm, repo = FakeLlm(), InMemoryConversationRepository()
    scripted_turn(llm, "Liều paracetamol")
    service = service_with(llm, repo)
    empty = Conversation.create_empty(user_id=OWNER, now=NOW)
    empty.rename("Hỏi về thuốc hạ sốt", now=NOW)
    await repo.create(empty)

    await service.ask(
        user_id=OWNER, message="Paracetamol?", conversation_id=empty.conversation_id
    )

    assert repo.rows[empty.conversation_id].title == "Hỏi về thuốc hạ sốt"
