import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.engine import make_url

from pharma_agent.infrastructure.auth.session_cleanup import (
    delete_expired_access_tokens,
)
from pharma_agent.infrastructure.persistence.postgres.database import Database
from pharma_agent.infrastructure.persistence.postgres.tables import (
    AccessTokenTable,
    UserTable,
)

pytestmark = pytest.mark.integration

LIFETIME = 7 * 24 * 3600


@pytest.fixture
async def database(migrated_dsn: str) -> AsyncGenerator[Database]:
    db = Database(migrated_dsn, pool_size=2)
    async with db.engine.begin() as connection:
        await connection.execute(text("TRUNCATE access_tokens"))
    yield db
    await db.dispose()


async def test_only_expired_sessions_are_deleted(
    database: Database, migrated_dsn: str
) -> None:
    user_id = uuid.uuid4()
    now = datetime.now(UTC)
    async with database.sessions.begin() as session:
        session.add(
            UserTable(
                id=user_id,
                email=f"{user_id.hex[:12]}@example.com",
                hashed_password="x",
            )
        )
        await session.flush()
        session.add_all(
            [
                AccessTokenTable(
                    token="expired-token",
                    user_id=user_id,
                    created_at=now - timedelta(seconds=LIFETIME + 60),
                ),
                AccessTokenTable(
                    token="fresh-token",
                    user_id=user_id,
                    created_at=now - timedelta(seconds=LIFETIME - 60),
                ),
            ]
        )
    conninfo = (
        make_url(migrated_dsn)
        .set(drivername="postgresql")
        .render_as_string(hide_password=False)
    )

    assert await delete_expired_access_tokens(conninfo, lifetime_seconds=LIFETIME) == 1

    async with database.sessions() as session:
        remaining = (
            (await session.execute(text("SELECT token FROM access_tokens")))
            .scalars()
            .all()
        )
    assert remaining == ["fresh-token"]
    assert await delete_expired_access_tokens(conninfo, lifetime_seconds=LIFETIME) == 0
