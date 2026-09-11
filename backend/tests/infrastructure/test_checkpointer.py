import warnings

import pytest
from sqlalchemy.engine import make_url

from pharma_agent.application.chat.graph import build_chat_graph
from pharma_agent.application.chat.runner import ChatTurnRunner
from pharma_agent.domain.agent.budget import BudgetLimits
from pharma_agent.domain.agent.run import AgentRun
from pharma_agent.domain.agent.schemas import JudgeDecision, JudgeOutcome
from pharma_agent.domain.llm.models import LlmRole
from pharma_agent.infrastructure.langgraph.checkpointer import (
    open_postgres_checkpointer,
)
from tests.application.test_chat_graph import QUESTION, passing_llm
from tests.domain.factories import make_hit
from tests.fakes import FakeRetriever, build_deps

pytestmark = pytest.mark.integration


async def test_graph_state_is_checkpointed_in_postgres(fresh_database_dsn: str) -> None:
    conninfo = (
        make_url(fresh_database_dsn)
        .set(drivername="postgresql")
        .render_as_string(hide_password=False)
    )
    llm = passing_llm()
    llm.script(
        LlmRole.JUDGE, JudgeDecision(decision=JudgeOutcome.ANSWER, gaps=[], reason="đủ")
    )
    async with open_postgres_checkpointer(conninfo, max_size=2) as saver:
        graph = build_chat_graph(checkpointer=saver)
        runner = ChatTurnRunner(
            graph,
            build_deps(llm, FakeRetriever([make_hit("c1", fusion=0.9)])),
            BudgetLimits(),
        )
        execution = runner.start(user_id="u1", message=QUESTION)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            async for _ in execution.events():
                pass
            snapshot = await graph.aget_state(
                {"configurable": {"thread_id": execution.run.run_id}}
            )
    run = (
        snapshot.values["run"]
        if isinstance(snapshot.values, dict)
        else snapshot.values.run
    )
    assert isinstance(run, AgentRun) and run.status.value == "completed"
