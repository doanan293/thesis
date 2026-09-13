import asyncio

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
from pharma_agent.infrastructure.persistence.postgres.corpus_tables import (
    CORPUS_SCHEMA,
)
from pharma_agent.infrastructure.persistence.postgres.metadata import (
    include_name,
    target_metadata,
)

pytestmark = pytest.mark.integration

EXPECTED_TABLES = {
    "user",
    "oauth_account",
    "conversations",
    "messages",
    "retrieval_runs",
    "retrieval_hits",
}
EXPECTED_CORPUS_TABLES = {
    "collections",
    "documents",
    "sections",
    "section_revisions",
    "chunk_versions",
    "releases",
    "release_chunks",
    "glossary_entries",
    "colloquial_mappings",
    "embedding_cache",
}


def _diff(connection: Connection) -> list[object]:
    context = MigrationContext.configure(
        connection,
        opts={
            "compare_type": True,
            "include_schemas": True,
            "include_name": include_name,
        },
    )
    return list(compare_metadata(context, target_metadata))


def _tables(connection: Connection, schema: str | None = None) -> set[str]:
    return set(inspect(connection).get_table_names(schema=schema))


def _schemas(connection: Connection) -> set[str]:
    return set(inspect(connection).get_schema_names())


async def test_upgrade_head_matches_the_models(migrated_dsn: str) -> None:
    engine = create_async_engine(migrated_dsn)
    async with engine.connect() as connection:
        assert await connection.run_sync(_tables) >= EXPECTED_TABLES
        assert (
            await connection.run_sync(_tables, CORPUS_SCHEMA) == EXPECTED_CORPUS_TABLES
        )
        assert await connection.run_sync(_diff) == []
    await engine.dispose()


async def test_downgrade_then_upgrade_round_trips(fresh_database_dsn: str) -> None:
    config = alembic_config(fresh_database_dsn)
    await asyncio.to_thread(command.upgrade, config, "head")
    await asyncio.to_thread(command.downgrade, config, "base")
    engine = create_async_engine(fresh_database_dsn)
    async with engine.connect() as connection:
        assert not (EXPECTED_TABLES & await connection.run_sync(_tables))
        assert CORPUS_SCHEMA not in await connection.run_sync(_schemas)
    await engine.dispose()
    await asyncio.to_thread(command.upgrade, config, "head")
