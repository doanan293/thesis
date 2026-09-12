from collections.abc import Generator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from pharma_agent.application.chat.graph import build_chat_graph
from pharma_agent.application.chat.runner import ChatTurnRunner
from pharma_agent.application.tracing import NullTracing, TraceHandle
from pharma_agent.domain.agent.budget import BudgetLimits
from pharma_agent.domain.agent.run import AgentRun
from pharma_agent.domain.agent.schemas import JudgeDecision, JudgeOutcome
from pharma_agent.domain.llm.models import LlmRole
from tests.application.test_chat_graph import QUESTION, passing_llm
from tests.domain.factories import make_hit
from tests.fakes import FakeRetriever, build_deps


class CountingHandler(BaseCallbackHandler):
    run_inline = True

    def __init__(self) -> None:
        super().__init__()
        self.chain_starts = 0

    def on_chain_start(
        self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any
    ) -> None:
        self.chain_starts += 1


class RecordingTracer:
    def __init__(self) -> None:
        self.handler = CountingHandler()
        self.opened: list[tuple[str, str, str | None]] = []
        self.finished: list[tuple[str, str]] = []

    @contextmanager
    def trace_turn(self, run: AgentRun) -> Generator[TraceHandle]:
        self.opened.append((run.run_id, run.user_id, run.conversation_id))
        yield _RecordingHandle(self)


class _RecordingHandle:
    def __init__(self, tracer: RecordingTracer) -> None:
        self._tracer = tracer

    @property
    def callbacks(self) -> Sequence[BaseCallbackHandler]:
        return (self._tracer.handler,)

    def finish(self, *, status: str, output: str) -> None:
        self._tracer.finished.append((status, output))


async def test_runner_traces_each_turn_and_forwards_callbacks() -> None:
    llm = passing_llm()
    llm.script(
        LlmRole.JUDGE, JudgeDecision(decision=JudgeOutcome.ANSWER, gaps=[], reason="đủ")
    )
    tracer = RecordingTracer()
    runner = ChatTurnRunner(
        build_chat_graph(),
        build_deps(llm, FakeRetriever([make_hit("c1", fusion=0.9)])),
        BudgetLimits(),
        tracer=tracer,
    )
    execution = runner.start(user_id="u1", message=QUESTION, conversation_id="c" * 32)
    async for _ in execution.events():
        pass

    assert execution.outcome is not None
    assert tracer.opened == [(execution.run.run_id, "u1", "c" * 32)]
    assert tracer.finished == [("completed", execution.outcome.answer_text)]
    assert tracer.handler.chain_starts >= 1


async def test_null_tracing_is_a_no_op() -> None:
    tracing = NullTracing()
    with tracing.trace_turn(
        AgentRun.start(
            user_id="u",
            original_query="q",
            limits=BudgetLimits(),
            now=datetime.now(UTC),
        )
    ) as handle:
        assert handle.callbacks == ()
        handle.finish(status="completed", output="x")
    tracing.record_feedback(run_id="r", rating="up", note="")
