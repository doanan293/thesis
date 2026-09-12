import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection, sql
from sqlalchemy.engine import make_url

from pharma_agent.application.chat.graph import build_chat_graph
from pharma_agent.application.chat.runner import ChatTurnRunner
from pharma_agent.domain.agent.budget import BudgetLimits
from pharma_agent.domain.agent.schemas import JudgeDecision, JudgeOutcome
from pharma_agent.domain.llm.models import LlmRole
from pharma_agent.infrastructure.langgraph.checkpointer import (
    open_postgres_checkpointer,
)
from pharma_agent.infrastructure.langgraph.cleanup import delete_expired_checkpoints
from tests.application.test_chat_graph import QUESTION, passing_llm
from tests.domain.factories import make_hit
from tests.fakes import FakeRetriever, build_deps

pytestmark = pytest.mark.integration

TABLES = ("checkpoints", "checkpoint_writes", "checkpoint_blobs")


async def run_turn(saver: AsyncPostgresSaver) -> str:
    llm = passing_llm()
    llm.script(
        LlmRole.JUDGE, JudgeDecision(decision=JudgeOutcome.ANSWER, gaps=[], reason="đủ")
    )
    runner = ChatTurnRunner(
        build_chat_graph(checkpointer=saver),
        build_deps(llm, FakeRetriever([make_hit("c1", fusion=0.9)])),
        BudgetLimits(),
    )
    execution = runner.start(user_id="u1", message=QUESTION)
    async for _ in execution.events():
        pass
    return execution.run.run_id


async def count_rows(conninfo: str, table: str, thread_id: str) -> int:
    query = sql.SQL("SELECT count(*) FROM {} WHERE thread_id = %s").format(
        sql.Identifier(table)
    )
    async with await AsyncConnection.connect(conninfo) as connection:
        cursor = await connection.execute(query, (thread_id,))
        row = await cursor.fetchone()
    return int(row[0]) if row else 0


async def test_only_expired_threads_are_removed(fresh_database_dsn: str) -> None:
    conninfo = (
        make_url(fresh_database_dsn)
        .set(drivername="postgresql")
        .render_as_string(hide_password=False)
    )
    async with open_postgres_checkpointer(conninfo, max_size=2) as saver:
        old_thread = await run_turn(saver)
        fresh_thread = await run_turn(saver)

    old_checkpoints = await count_rows(conninfo, "checkpoints", old_thread)
    assert old_checkpoints > 0
    assert await count_rows(conninfo, "checkpoint_blobs", old_thread) > 0
    async with await AsyncConnection.connect(conninfo, autocommit=True) as connection:
        await connection.execute(
            "UPDATE checkpoints SET checkpoint = jsonb_set("
            "checkpoint, '{ts}', to_jsonb((now() - interval '10 days')::text)) "
            "WHERE thread_id = %s",
            (old_thread,),
        )

    deleted = await delete_expired_checkpoints(conninfo, retention_days=7)

    assert deleted == old_checkpoints
    for table in TABLES:
        assert await count_rows(conninfo, table, old_thread) == 0, table
    assert await count_rows(conninfo, "checkpoints", fresh_thread) > 0
    assert await count_rows(conninfo, "checkpoint_blobs", fresh_thread) > 0
    assert await delete_expired_checkpoints(conninfo, retention_days=7) == 0
