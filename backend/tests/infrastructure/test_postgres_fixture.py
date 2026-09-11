import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

pytestmark = pytest.mark.integration


async def test_postgres_fixture_is_reachable(postgres_dsn: str) -> None:
    assert postgres_dsn.startswith("postgresql+psycopg://")
    engine = create_async_engine(postgres_dsn)
    async with engine.connect() as connection:
        assert (await connection.execute(text("select 1"))).scalar_one() == 1
    await engine.dispose()
