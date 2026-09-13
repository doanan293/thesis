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


def _audit_column_names(connection: Connection, table: str) -> set[str]:
    return {column["name"] for column in inspect(connection).get_columns(table)}


def _audit_referred_tables(connection: Connection, table: str) -> set[str]:
    return {fk["referred_table"] for fk in inspect(connection).get_foreign_keys(table)}


async def test_audit_tables_reference_releases_and_chunk_versions(
    migrated_dsn: str,
) -> None:
    engine = create_async_engine(migrated_dsn)
    async with engine.connect() as connection:
        runs = await connection.run_sync(_audit_column_names, "retrieval_runs")
        hits = await connection.run_sync(_audit_column_names, "retrieval_hits")
        hit_references = await connection.run_sync(
            _audit_referred_tables, "retrieval_hits"
        )
    await engine.dispose()
    assert "release_ids" in runs and "corpus_version" not in runs
    assert {"chunk_version_id", "section_key", "table_key"} <= hits
    assert not {"chunk_id", "section_id", "table_id"} & hits
    assert hit_references == {"retrieval_runs"}  # never a FK into schema corpus


def _indexes(connection: Connection, table: str) -> dict[str | None, list[str | None]]:
    return {
        index["name"]: list(index["column_names"])
        for index in inspect(connection).get_indexes(table)
    }


async def test_keyset_indexes_cover_the_sort_keys(migrated_dsn: str) -> None:
    engine = create_async_engine(migrated_dsn)
    async with engine.connect() as connection:
        conversations = await connection.run_sync(_indexes, "conversations")
        messages = await connection.run_sync(_indexes, "messages")
    await engine.dispose()
    assert conversations["ix_conversations_user_updated_id"] == [
        "user_id",
        "updated_at",
        "id",
    ]
    assert "ix_conversations_user_updated" not in conversations
    assert messages["ix_messages_conversation_created_id"] == [
        "conversation_id",
        "created_at",
        "id",
    ]
    assert "ix_messages_conversation_created" not in messages
