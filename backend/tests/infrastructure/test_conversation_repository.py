import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timedelta

import pytest
from sqlalchemy import func, select, text

from pharma_agent.domain.conversation.models import (
    Conversation,
    Message,
    MessageRole,
)
from pharma_agent.domain.retrieval.audit import RetrievalHitRecord, RetrievalRunRecord
from pharma_agent.infrastructure.persistence.postgres.conversation_repository import (
    AuditContext,
    ConversationRowMissing,
    PostgresConversationRepository,
)
from pharma_agent.infrastructure.persistence.postgres.database import Database
from pharma_agent.infrastructure.persistence.postgres.tables import (
    RetrievalHitTable,
    RetrievalRunTable,
    UserTable,
)
from tests.domain.factories import (
    COLLECTION_ID,
    NOW,
    RELEASE_ID,
    SECTION_KEY,
    chunk_uuid,
    make_citation,
)

pytestmark = pytest.mark.integration


@pytest.fixture
async def database(migrated_dsn: str) -> AsyncIterator[Database]:
    db = Database(migrated_dsn, pool_size=2)
    async with db.engine.begin() as connection:
        await connection.execute(
            text(
                'TRUNCATE "user", conversations, messages, retrieval_runs, retrieval_hits CASCADE'
            )
        )
    yield db
    await db.dispose()


async def make_user(database: Database, email: str = "a@example.com") -> str:
    user_id = uuid.uuid4()
    async with database.sessions.begin() as session:
        session.add(UserTable(id=user_id, email=email, hashed_password="x"))
    return user_id.hex


def repository(database: Database) -> PostgresConversationRepository:
    return PostgresConversationRepository(
        database.sessions,
        AuditContext(
            embedding_model="qwen3-embedding:4b-fp16",
            retriever_config={"mode": "hybrid", "rrf_k": 2},
        ),
    )


def turn(
    conversation_id: str, index: int, status: str = "completed"
) -> tuple[Message, Message]:
    at = NOW + timedelta(minutes=index)
    citation = make_citation("c1")
    return (
        Message(
            message_id=uuid.uuid4().hex,
            conversation_id=conversation_id,
            role=MessageRole.USER,
            content=f"q{index}",
            status=status,
            run_id="r" * 32,
            created_at=at,
        ),
        Message(
            message_id=uuid.uuid4().hex,
            conversation_id=conversation_id,
            role=MessageRole.ASSISTANT,
            content=f"a{index}",
            status=status,
            citations=[citation],
            phases=["answering"],
            usage={"llm_calls": 5},
            run_id="r" * 32,
            created_at=at + timedelta(microseconds=1),
        ),
    )


AUDIT = [
    RetrievalRunRecord(
        round=1,
        query_text="paracetamol liều",
        release_ids={str(COLLECTION_ID): str(RELEASE_ID)},
        hits=[
            RetrievalHitRecord(
                rank=1,
                chunk_version_id=chunk_uuid("c1"),
                section_key=SECTION_KEY,
                table_key=None,
                fusion_score=0.5,
                rerank_score=0.9,
                hydrate_strategy="chunk_window",
                cited=True,
                snippet="x",
            ),
        ],
    )
]


async def test_create_get_is_scoped_to_owner(database: Database) -> None:
    repo = repository(database)
    owner = await make_user(database)
    stranger = await make_user(database, "b@example.com")
    conversation = Conversation.start(
        user_id=owner, first_message="Paracetamol?", now=NOW
    )
    await repo.create(conversation)

    loaded = await repo.get(owner, conversation.conversation_id)
    assert (
        loaded is not None
        and loaded.title == "Paracetamol?"
        and loaded.created_at == NOW
    )
    assert await repo.get(stranger, conversation.conversation_id) is None
    assert await repo.get(owner, "not-a-uuid") is None
    assert await repo.delete(stranger, conversation.conversation_id) is False


async def test_append_turn_is_atomic_and_increments_turn_count(
    database: Database,
) -> None:
    repo = repository(database)
    owner = await make_user(database)
    conversation = Conversation.start(user_id=owner, first_message="hi", now=NOW)
    await repo.create(conversation)

    for index in range(3):
        user_msg, assistant_msg = turn(conversation.conversation_id, index)
        conversation.record_turn(assistant_msg.created_at)
        await repo.append_turn(
            conversation, user_msg, assistant_msg, AUDIT if index == 0 else []
        )

    loaded = await repo.get(owner, conversation.conversation_id)
    assert loaded is not None and loaded.turn_count == 3
    assert [
        t.user_text for t in await repo.recent_turns(conversation.conversation_id, 2)
    ] == ["q1", "q2"]
    assert [
        t.user_text for t in await repo.turns_since(conversation.conversation_id, 1)
    ] == ["q1", "q2"]
    messages = await repo.messages(conversation.conversation_id, limit=3)
    assert [m.content for m in messages] == ["a1", "q2", "a2"]
    assert messages[0].citations[0].chunk_version_id == chunk_uuid("c1") and messages[
        0
    ].usage == {"llm_calls": 5}
    older = await repo.messages(
        conversation.conversation_id,
        limit=10,
        cursor=(messages[0].created_at, messages[0].message_id),
    )
    assert [m.content for m in older] == ["q0", "a0", "q1"]

    async with database.sessions() as session:
        run = (await session.execute(select(RetrievalRunTable))).scalar_one()
        assert run.release_ids == {str(COLLECTION_ID): str(RELEASE_ID)}
        assert run.round == 1 and run.embedding_model == "qwen3-embedding:4b-fp16"
        hit = (await session.execute(select(RetrievalHitTable))).scalar_one()
        assert hit.cited is True and hit.rerank_score == 0.9
        assert (hit.chunk_version_id, hit.section_key, hit.table_key) == (
            chunk_uuid("c1"),
            SECTION_KEY,
            None,
        )


async def test_append_turn_rolls_back_when_conversation_is_missing(
    database: Database,
) -> None:
    repo = repository(database)
    owner = await make_user(database)
    ghost = Conversation.start(user_id=owner, first_message="hi", now=NOW)
    user_msg, assistant_msg = turn(ghost.conversation_id, 0)
    with pytest.raises(ConversationRowMissing):
        await repo.append_turn(ghost, user_msg, assistant_msg, AUDIT)
    async with database.sessions() as session:
        assert (
            await session.execute(select(func.count()).select_from(RetrievalRunTable))
        ).scalar_one() == 0


async def test_update_summary_does_not_overwrite_turn_count(database: Database) -> None:
    repo = repository(database)
    owner = await make_user(database)
    conversation = Conversation.start(user_id=owner, first_message="hi", now=NOW)
    await repo.create(conversation)
    stale = conversation.model_copy(deep=True)

    for index in range(2):
        user_msg, assistant_msg = turn(conversation.conversation_id, index)
        conversation.record_turn(assistant_msg.created_at)
        await repo.append_turn(conversation, user_msg, assistant_msg, [])

    stale.turn_count = 2
    stale.apply_summary("tóm tắt", covered_turns=2, now=NOW + timedelta(hours=1))
    await repo.update_summary(stale)
    stale.rename("Tên mới", now=NOW + timedelta(hours=1))
    await repo.update_title(stale)

    loaded = await repo.get(owner, conversation.conversation_id)
    assert loaded is not None
    assert (
        loaded.turn_count,
        loaded.summarized_turns,
        loaded.summary,
        loaded.title,
    ) == (2, 2, "tóm tắt", "Tên mới")


async def test_list_for_user_orders_by_recent_activity_and_deletes_cascade(
    database: Database,
) -> None:
    repo = repository(database)
    owner = await make_user(database)
    first = Conversation.start(user_id=owner, first_message="first", now=NOW)
    second = Conversation.start(
        user_id=owner, first_message="second", now=NOW + timedelta(minutes=5)
    )
    empty = Conversation.start(
        user_id=owner, first_message="empty", now=NOW + timedelta(minutes=30)
    )
    for conversation in (first, second, empty):
        await repo.create(conversation)
    for conversation, index in ((second, 7), (first, 10)):
        user_msg, assistant_msg = turn(conversation.conversation_id, index)
        conversation.record_turn(assistant_msg.created_at)
        await repo.append_turn(
            conversation,
            user_msg,
            assistant_msg,
            AUDIT if conversation is first else [],
        )

    listed = await repo.list_for_user(owner, limit=10)
    assert [c.title for c in listed] == ["first", "second"]
    after_first = (listed[0].updated_at, listed[0].conversation_id)
    assert [
        c.title for c in await repo.list_for_user(owner, limit=10, cursor=after_first)
    ] == ["second"]

    assert await repo.delete(owner, first.conversation_id) is True
    async with database.sessions() as session:
        assert (
            await session.execute(select(func.count()).select_from(RetrievalHitTable))
        ).scalar_one() == 0


def tied_turn(conversation_id: str) -> tuple[Message, Message]:
    """Both messages share NOW, so only the id breaks the tie."""
    return (
        Message(
            message_id=uuid.uuid4().hex,
            conversation_id=conversation_id,
            role=MessageRole.USER,
            content="q",
            status="completed",
            created_at=NOW,
        ),
        Message(
            message_id=uuid.uuid4().hex,
            conversation_id=conversation_id,
            role=MessageRole.ASSISTANT,
            content="a",
            status="completed",
            created_at=NOW,
        ),
    )


async def test_keyset_pages_have_no_duplicates_or_gaps_with_tied_timestamps(
    database: Database,
) -> None:
    repo = repository(database)
    owner = await make_user(database)
    created: list[Conversation] = []
    for index in range(5):
        conversation = Conversation.start(
            user_id=owner, first_message=f"c{index}", now=NOW
        )
        await repo.create(conversation)
        user_msg, assistant_msg = tied_turn(conversation.conversation_id)
        conversation.record_turn(NOW)
        await repo.append_turn(conversation, user_msg, assistant_msg, [])
        created.append(conversation)
    await repo.create(
        Conversation.start(
            user_id=owner, first_message="empty", now=NOW + timedelta(hours=1)
        )
    )

    seen: list[str] = []
    position: tuple[datetime, str] | None = None
    while page := await repo.list_for_user(owner, limit=2, cursor=position):
        seen.extend(c.conversation_id for c in page)
        position = (page[-1].updated_at, page[-1].conversation_id)
    assert seen == sorted((c.conversation_id for c in created), reverse=True)

    target = created[0]
    for _ in range(3):
        user_msg, assistant_msg = tied_turn(target.conversation_id)
        target.record_turn(NOW)
        await repo.append_turn(target, user_msg, assistant_msg, [])
    everything = [
        m.message_id for m in await repo.messages(target.conversation_id, limit=100)
    ]
    assert everything == sorted(everything) and len(set(everything)) == 8

    pages: list[list[str]] = []
    position = None
    while page := await repo.messages(target.conversation_id, limit=3, cursor=position):
        ids = [m.message_id for m in page]
        assert ids == sorted(ids)
        pages.append(ids)
        position = (page[0].created_at, page[0].message_id)
    assert [len(ids) for ids in pages] == [3, 3, 2]
    assert [mid for ids in reversed(pages) for mid in ids] == everything
