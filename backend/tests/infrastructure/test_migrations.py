import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import inspect
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from pharma_agent.infrastructure.persistence.postgres.alembic_config import (
    alembic_config,
)
from pharma_agent.infrastructure.persistence.postgres.tables import Base, include_name

pytestmark = pytest.mark.integration

EXPECTED_TABLES = {
    "user",
    "oauth_account",
    "conversations",
    "messages",
    "retrieval_runs",
    "retrieval_hits",
}


def _diff(connection: Connection) -> list[object]:
    context = MigrationContext.configure(
        connection, opts={"compare_type": True, "include_name": include_name}
    )
    return list(compare_metadata(context, Base.metadata))


def _tables(connection: Connection) -> set[str]:
    return set(inspect(connection).get_table_names())


async def test_upgrade_head_matches_the_models(migrated_dsn: str) -> None:
    engine = create_async_engine(migrated_dsn)
    async with engine.connect() as connection:
        assert await connection.run_sync(_tables) >= EXPECTED_TABLES
        assert await connection.run_sync(_diff) == []
    await engine.dispose()


async def test_downgrade_then_upgrade_round_trips(fresh_database_dsn: str) -> None:
    import asyncio

    config = alembic_config(fresh_database_dsn)
    await asyncio.to_thread(command.upgrade, config, "head")
    await asyncio.to_thread(command.downgrade, config, "base")
    engine = create_async_engine(fresh_database_dsn)
    async with engine.connect() as connection:
        assert not (EXPECTED_TABLES & await connection.run_sync(_tables))
    await engine.dispose()
    await asyncio.to_thread(command.upgrade, config, "head")
