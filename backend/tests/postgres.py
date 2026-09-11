"""Session-scoped Postgres for integration tests (one container per test session)."""

from collections.abc import Iterator

import pytest


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
