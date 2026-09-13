import uuid
from collections.abc import AsyncGenerator

import pytest
from sqlalchemy import text, update

from pharma_agent.infrastructure.persistence.postgres.citation_reader import (
    PostgresCitationReader,
)
from pharma_agent.infrastructure.persistence.postgres.corpus_tables import (
    CollectionTable,
)
from pharma_agent.infrastructure.persistence.postgres.database import Database
from tests.corpus_rows import add_release, seed_cited_release

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
