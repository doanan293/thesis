import uuid
from collections.abc import AsyncGenerator

import pytest
from sqlalchemy import text, update

from pharma_agent.domain.conversation.models import Conversation
from pharma_agent.infrastructure.persistence.postgres.citation_reader import (
    PostgresCitationReader,
)
from pharma_agent.infrastructure.persistence.postgres.corpus_tables import (
    CollectionTable,
)
from pharma_agent.infrastructure.persistence.postgres.database import Database
from tests.corpus_rows import (
    add_release,
    citation_for,
    dosage_drafts,
    seed_cited_release,
)
from tests.domain.factories import NOW
from tests.infrastructure.test_conversation_repository import (
    make_user,
    repository,
    turn,
)

pytestmark = pytest.mark.integration


@pytest.fixture
async def database(migrated_dsn: str) -> AsyncGenerator[Database]:
    db = Database(migrated_dsn, pool_size=2)
    async with db.engine.begin() as connection:
        await connection.execute(
            text('TRUNCATE "user", conversations, messages, message_citations CASCADE')
        )
    yield db
    await db.dispose()


async def test_current_release_ids_follow_collections_current_release(
    database: Database,
) -> None:
    reader = PostgresCitationReader(database.sessions)
    seeded = await seed_cited_release(database.sessions)
    other = await add_release(database.sessions, seeded.collection_id, number=2)
    unknown = uuid.uuid4()

    assert await reader.current_release_ids({seeded.release_id, other, unknown}) == {
        seeded.release_id
    }
    assert await reader.current_release_ids(set()) == set()

    async with database.sessions.begin() as session:
        await session.execute(
            update(CollectionTable)
            .where(CollectionTable.id == seeded.collection_id)
            .values(current_release_id=other)
        )
    assert await reader.current_release_ids([seeded.release_id, other]) == {other}


async def test_citation_block_keeps_reading_order_and_is_owner_scoped(
    database: Database,
) -> None:
    seeded = await seed_cited_release(database.sessions)
    drafts = dosage_drafts(seeded)
    order = (2, 0, 1)
    repo = repository(database)
    owner = await make_user(database)
    stranger = await make_user(database, "b@example.com")
    conversation = Conversation.start(user_id=owner, first_message="hi", now=NOW)
    await repo.create(conversation)
    cited = citation_for(seeded, index=1, position=1, block=order)
    user_msg, assistant_msg = turn(conversation.conversation_id, 0, citations=[cited])
    conversation.record_turn(assistant_msg.created_at)
    await repo.append_turn(conversation, user_msg, assistant_msg, [])
    reader = PostgresCitationReader(database.sessions)

    block = await reader.citation_block(owner, assistant_msg.message_id, 1)

    assert block is not None and block.citation == cited and block.is_current is True
    assert [chunk.chunk_version_id for chunk in block.chunks] == [
        drafts[n].chunk_version_id for n in order
    ]
    assert [chunk.text for chunk in block.chunks] == [
        drafts[n].chunk_text for n in order
    ]
    assert [(chunk.start_page, chunk.end_page) for chunk in block.chunks] == [
        (drafts[n].start_page, drafts[n].end_page) for n in order
    ]
    assert await reader.citation_block(stranger, assistant_msg.message_id, 1) is None
    assert await reader.citation_block(owner, assistant_msg.message_id, 2) is None
    assert await reader.citation_block(owner, "not-a-uuid", 1) is None

    newer = await add_release(database.sessions, seeded.collection_id, number=2)
    async with database.sessions.begin() as session:
        await session.execute(
            update(CollectionTable)
            .where(CollectionTable.id == seeded.collection_id)
            .values(current_release_id=newer)
        )
    moved = await reader.citation_block(owner, assistant_msg.message_id, 1)
    assert moved is not None and moved.is_current is False
