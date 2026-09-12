"""Ports for observability: one trace per chat turn, and scores attached to a turn later."""

from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import Protocol

from langchain_core.callbacks import BaseCallbackHandler

from pharma_agent.domain.agent.run import AgentRun


class TraceHandle(Protocol):
    @property
    def callbacks(self) -> Sequence[BaseCallbackHandler]:
        """LangChain callbacks to attach to the graph run so node spans join the trace."""
        ...

    def finish(self, *, status: str, output: str) -> None: ...


class TurnTracer(Protocol):
    def trace_turn(self, run: AgentRun) -> AbstractContextManager[TraceHandle]: ...


class ScoreSink(Protocol):
    def record_feedback(self, *, run_id: str, rating: str, note: str) -> None:
        """Attach a user rating to the trace of the turn identified by run_id."""
        ...


class Tracing(TurnTracer, ScoreSink, Protocol):
    """A tracer that can also score its own traces (Langfuse, or the null implementation)."""


class _NullHandle:
    callbacks: Sequence[BaseCallbackHandler] = ()

    def finish(self, *, status: str, output: str) -> None:
        return None


class NullTracing:
    """Used when Langfuse is not configured."""

    def trace_turn(self, run: AgentRun) -> AbstractContextManager[TraceHandle]:
        return nullcontext(_NullHandle())

    def record_feedback(self, *, run_id: str, rating: str, note: str) -> None:
        return None
