from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

from pharma_agent.application.chat.checkpoint import checkpoint_serializer


@asynccontextmanager
async def open_postgres_checkpointer(
    conninfo: str, *, max_size: int = 10
) -> AsyncGenerator[AsyncPostgresSaver]:
    """Pooled AsyncPostgresSaver with the strict msgpack allowlist; creates its tables on start."""
    pool: AsyncConnectionPool[AsyncConnection[DictRow]] = AsyncConnectionPool(
        conninfo,
        connection_class=AsyncConnection[DictRow],
        min_size=1,
        max_size=max_size,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    async with pool:
        saver = AsyncPostgresSaver(pool, serde=checkpoint_serializer())
        await saver.setup()
        yield saver
