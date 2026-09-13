from datetime import timedelta

from pharma_agent.application.memory.summarize import SummarizeConversation
from pharma_agent.domain.conversation.models import Conversation, ConversationSummary
from pharma_agent.domain.conversation.turns import build_turn_messages
from pharma_agent.domain.llm.models import LlmRole
from pharma_agent.domain.llm.port import LlmError
from pharma_agent.domain.shared.clock import FixedClock
from pharma_agent.domain.shared.ids import new_id
from tests.domain.factories import make_run
from tests.fakes import NOW, FakeLlm
from tests.memory_repository import InMemoryConversationRepository

OWNER = "a" * 32


async def seeded(
    repo: InMemoryConversationRepository, statuses: list[str]
) -> Conversation:
    conversation = Conversation.start(user_id=OWNER, first_message="hi", now=NOW)
    await repo.create(conversation)
    for index, status in enumerate(statuses):
        run = make_run(f"câu hỏi {index}")
        user_msg, assistant_msg = build_turn_messages(
            user_message_id=new_id(),
            assistant_message_id=new_id(),
            conversation_id=conversation.conversation_id,
            run=run,
            answer_text=f"trả lời {index}",
            citations=[],
            phases=[],
            now=NOW + timedelta(minutes=index),
        )
        user_msg = user_msg.model_copy(update={"status": status})
        assistant_msg = assistant_msg.model_copy(update={"status": status})
        conversation.record_turn(assistant_msg.created_at)
        await repo.append_turn(conversation, user_msg, assistant_msg, [])
    return conversation


def summarizer(
    llm: FakeLlm, repo: InMemoryConversationRepository
) -> SummarizeConversation:
    return SummarizeConversation(
        llm, repo, FixedClock(NOW + timedelta(hours=1)), every=2, max_chars=20
    )


async def test_not_due_does_not_call_llm() -> None:
    llm, repo = FakeLlm(), InMemoryConversationRepository()
    conversation = await seeded(repo, ["completed"])
    assert (
        await summarizer(llm, repo).run_if_needed(
            user_id=OWNER, conversation_id=conversation.conversation_id
        )
        is False
    )
    assert llm.calls == []


async def test_due_summary_uses_usable_turns_and_is_truncated() -> None:
    llm, repo = FakeLlm(), InMemoryConversationRepository()
    conversation = await seeded(repo, ["completed", "blocked"])
    llm.script(
        LlmRole.SUMMARIZER,
        ConversationSummary(summary="Người dùng hỏi về paracetamol và liều dùng."),
    )

    assert (
        await summarizer(llm, repo).run_if_needed(
            user_id=OWNER, conversation_id=conversation.conversation_id
        )
        is True
    )

    prompt = llm.calls_for(LlmRole.SUMMARIZER)[0][1].content
    assert "câu hỏi 0" in prompt and "câu hỏi 1" not in prompt
    stored = repo.rows[conversation.conversation_id]
    assert stored.summarized_turns == 2 and stored.summary == "Người dùng hỏi về pa"
    assert stored.turn_count == 2


async def test_only_excluded_turns_advance_without_llm() -> None:
    llm, repo = FakeLlm(), InMemoryConversationRepository()
    conversation = await seeded(repo, ["error", "timeout"])
    assert (
        await summarizer(llm, repo).run_if_needed(
            user_id=OWNER, conversation_id=conversation.conversation_id
        )
        is False
    )
    assert (
        llm.calls == []
        and repo.rows[conversation.conversation_id].summarized_turns == 2
    )


async def test_llm_failure_keeps_previous_summary() -> None:
    llm, repo = FakeLlm(), InMemoryConversationRepository()
    conversation = await seeded(repo, ["completed", "completed"])
    llm.script(LlmRole.SUMMARIZER, LlmError("down"))
    assert (
        await summarizer(llm, repo).run_if_needed(
            user_id=OWNER, conversation_id=conversation.conversation_id
        )
        is False
    )
    assert repo.rows[conversation.conversation_id].summarized_turns == 0


async def test_missing_conversation_is_ignored() -> None:
    llm, repo = FakeLlm(), InMemoryConversationRepository()
    assert (
        await summarizer(llm, repo).run_if_needed(
            user_id=OWNER, conversation_id="f" * 32
        )
        is False
    )
