"""Postgres for integration tests: one container per session, databases created on demand."""

from collections.abc import Iterator
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from sqlalchemy.engine import make_url


@pytest.fixture(scope="session")
def postgres_dsn() -> Iterator[str]:
    from testcontainers.community.postgres import PostgresContainer

    with PostgresContainer(
        "postgres:17-alpine",
        username="thesis",
        password="thesis",
        dbname="thesis",
        driver="psycopg",
    ) as container:
        yield container.get_connection_url()


def create_database(server_dsn: str, name: str) -> str:
    url = make_url(server_dsn)
    conninfo = url.set(drivername="postgresql").render_as_string(hide_password=False)
    with psycopg.connect(conninfo, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    return url.set(database=name).render_as_string(hide_password=False)


@pytest.fixture
def fresh_database_dsn(postgres_dsn: str) -> str:
    return create_database(postgres_dsn, f"t_{uuid4().hex[:12]}")


@pytest.fixture(scope="session")
def migrated_dsn(postgres_dsn: str) -> str:
    from alembic import command

    from pharma_agent.infrastructure.persistence.postgres.alembic_config import (
        alembic_config,
    )

    dsn = create_database(postgres_dsn, "app")
    command.upgrade(alembic_config(dsn), "head")
    return dsn
