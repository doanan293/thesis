import uuid
from collections.abc import AsyncGenerator
from datetime import timedelta

import pytest
from sqlalchemy import text

from pharma_agent.domain.conversation.models import Conversation, Message, MessageRole
from pharma_agent.domain.feedback.models import Feedback, Rating
from pharma_agent.infrastructure.persistence.postgres.conversation_repository import (
    AuditContext,
    PostgresConversationRepository,
)
from pharma_agent.infrastructure.persistence.postgres.database import Database
from pharma_agent.infrastructure.persistence.postgres.feedback_repository import (
    PostgresFeedbackRepository,
)
from pharma_agent.infrastructure.persistence.postgres.tables import UserTable
from tests.domain.factories import NOW

pytestmark = pytest.mark.integration


@pytest.fixture
async def database(migrated_dsn: str) -> AsyncGenerator[Database]:
    db = Database(migrated_dsn, pool_size=2)
    async with db.engine.begin() as connection:
        await connection.execute(
            text('TRUNCATE "user", conversations, messages, feedback CASCADE')
        )
    yield db
    await db.dispose()


async def make_user(database: Database, email: str) -> str:
    user_id = uuid.uuid4()
    async with database.sessions.begin() as session:
        session.add(UserTable(id=user_id, email=email, hashed_password="x"))
    return user_id.hex


async def seed_turn(database: Database, owner: str) -> tuple[str, str]:
    conversations = PostgresConversationRepository(
        database.sessions, AuditContext(embedding_model="t")
    )
    conversation = Conversation.start(user_id=owner, first_message="hi", now=NOW)
    await conversations.create(conversation)
    user_msg = Message(
        message_id=uuid.uuid4().hex,
        conversation_id=conversation.conversation_id,
        role=MessageRole.USER,
        content="q",
        status="completed",
        created_at=NOW,
    )
    assistant_msg = Message(
        message_id=uuid.uuid4().hex,
        conversation_id=conversation.conversation_id,
        role=MessageRole.ASSISTANT,
        content="a",
        status="completed",
        run_id="r" * 32,
        created_at=NOW + timedelta(microseconds=1),
    )
    conversation.record_turn(NOW)
    await conversations.append_turn(conversation, user_msg, assistant_msg, [])
    return user_msg.message_id, assistant_msg.message_id


async def test_get_message_is_owner_scoped(database: Database) -> None:
    owner = await make_user(database, "a@example.com")
    stranger = await make_user(database, "b@example.com")
    user_id, assistant_id = await seed_turn(database, owner)
    conversations = PostgresConversationRepository(
        database.sessions, AuditContext(embedding_model="t")
    )
    found = await conversations.get_message(owner, assistant_id)
    assert (
        found is not None
        and found.role is MessageRole.ASSISTANT
        and found.run_id == "r" * 32
    )
    assert (await conversations.get_message(owner, user_id)) is not None
    assert await conversations.get_message(stranger, assistant_id) is None
    assert await conversations.get_message(owner, "nope") is None


async def test_save_upserts_per_user_and_message(database: Database) -> None:
    owner = await make_user(database, "a@example.com")
    _, assistant_id = await seed_turn(database, owner)
    repo = PostgresFeedbackRepository(database.sessions)
    first = Feedback.create(
        user_id=owner, message_id=assistant_id, rating=Rating.UP, note="ok", now=NOW
    )
    await repo.save(first)
    second = Feedback.create(
        user_id=owner,
        message_id=assistant_id,
        rating=Rating.DOWN,
        note="sai liều",
        now=NOW + timedelta(minutes=1),
    )
    await repo.save(second)
    stored = await repo.get(owner, assistant_id)
    assert stored is not None
    assert stored.feedback_id == first.feedback_id  # row identity is kept
    assert stored.rating is Rating.DOWN and stored.note == "sai liều"
    assert await repo.get("c" * 32, assistant_id) is None


async def test_for_messages_returns_only_this_users_feedback_in_one_call(
    database: Database,
) -> None:
    owner = await make_user(database, "a@example.com")
    stranger = await make_user(database, "b@example.com")
    _, first = await seed_turn(database, owner)
    _, second = await seed_turn(database, owner)
    repo = PostgresFeedbackRepository(database.sessions)
    mine = Feedback.create(
        user_id=owner, message_id=first, rating=Rating.DOWN, note="sai", now=NOW
    )
    theirs = Feedback.create(
        user_id=stranger, message_id=second, rating=Rating.UP, note="", now=NOW
    )
    await repo.save(mine)
    await repo.save(theirs)

    found = await repo.for_messages(owner, [first, second, "not-a-uuid"])

    assert found == {first: mine}
    assert await repo.for_messages(owner, []) == {}
