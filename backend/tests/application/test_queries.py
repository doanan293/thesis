from datetime import timedelta

import pytest

from pharma_agent.application.conversation.queries import ConversationQueries
from pharma_agent.application.errors import ConversationNotFound, InvalidInput
from pharma_agent.domain.conversation.models import Conversation
from pharma_agent.domain.conversation.turns import build_turn_messages
from pharma_agent.domain.shared.clock import FixedClock
from tests.domain.factories import make_run
from tests.fakes import NOW
from tests.memory_repository import InMemoryConversationRepository

OWNER, STRANGER = "a" * 32, "b" * 32


async def test_list_get_messages_rename_delete() -> None:
    repo = InMemoryConversationRepository()
    queries = ConversationQueries(repo, FixedClock(NOW + timedelta(days=1)))
    older = Conversation.start(user_id=OWNER, first_message="older", now=NOW)
    newer = Conversation.start(
        user_id=OWNER, first_message="newer", now=NOW + timedelta(minutes=1)
    )
    await repo.create(older)
    await repo.create(newer)
    user_msg, assistant_msg = build_turn_messages(
        conversation_id=older.conversation_id,
        run=make_run("q"),
        answer_text="a",
        citations=[],
        phases=["answering"],
        now=NOW,
    )
    older.record_turn(NOW + timedelta(minutes=2))
    await repo.append_turn(older, user_msg, assistant_msg, [])

    listed = await queries.list_conversations(OWNER, limit=10)
    assert [c.title for c in listed] == ["older", "newer"] and listed[0].turn_count == 1

    messages = await queries.list_messages(OWNER, older.conversation_id, limit=10)
    assert [(m.role, m.content) for m in messages] == [
        ("user", "q"),
        ("assistant", "a"),
    ]
    assert messages[1].phases == ["answering"]

    renamed = await queries.rename(OWNER, older.conversation_id, "  Thuốc hạ sốt ")
    assert renamed.title == "Thuốc hạ sốt" and renamed.updated_at == NOW + timedelta(
        days=1
    )

    with pytest.raises(InvalidInput):
        await queries.rename(OWNER, older.conversation_id, " ")
    for call in (
        queries.get_conversation(STRANGER, older.conversation_id),
        queries.list_messages(STRANGER, older.conversation_id, limit=10),
        queries.rename(STRANGER, older.conversation_id, "x"),
        queries.delete(STRANGER, older.conversation_id),
    ):
        with pytest.raises(ConversationNotFound):
            await call

    await queries.delete(OWNER, older.conversation_id)
    assert [c.title for c in await queries.list_conversations(OWNER, limit=10)] == [
        "newer"
    ]
