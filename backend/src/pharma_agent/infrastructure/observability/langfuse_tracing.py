"""Langfuse tracing: one trace per chat turn with a deterministic id derived from run_id,
LangGraph node spans nested inside it, and user feedback recorded as a score on it."""

import logging
from collections.abc import Generator, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager

from langchain_core.callbacks import BaseCallbackHandler
from langfuse import Langfuse, propagate_attributes
from langfuse._client.span import LangfuseAgent
from langfuse.langchain import CallbackHandler

from pharma_agent.application.tracing import TraceHandle
from pharma_agent.domain.agent.run import AgentRun
from pharma_agent.infrastructure.settings import LangfuseSettings

logger = logging.getLogger(__name__)
FEEDBACK_SCORE_NAME = "user_feedback"
TURN_OBSERVATION_NAME = "chat_turn"


class _Handle:
    def __init__(self, span: LangfuseAgent, handler: CallbackHandler) -> None:
        self._span = span
        self._callbacks: tuple[BaseCallbackHandler, ...] = (handler,)

    @property
    def callbacks(self) -> Sequence[BaseCallbackHandler]:
        return self._callbacks

    def finish(self, *, status: str, output: str) -> None:
        self._span.update(output=output, metadata={"status": status})


class LangfuseTracing:
    def __init__(self, client: Langfuse) -> None:
        self._client = client

    @classmethod
    def from_settings(cls, settings: LangfuseSettings) -> "LangfuseTracing":
        return cls(
            Langfuse(
                public_key=settings.public_key,
                secret_key=settings.secret_key,
                base_url=settings.host,
            )
        )

    def trace_id_for(self, run_id: str) -> str:
        return self._client.create_trace_id(seed=run_id)

    @contextmanager
    def _trace(self, run: AgentRun) -> Generator[TraceHandle]:
        with ExitStack() as stack:
            stack.enter_context(
                propagate_attributes(
                    user_id=run.user_id,
                    session_id=run.conversation_id,
                    trace_name=TURN_OBSERVATION_NAME,
                    metadata={"run_id": run.run_id},
                )
            )
            span = stack.enter_context(
                self._client.start_as_current_observation(
                    trace_context={"trace_id": self.trace_id_for(run.run_id)},
                    name=TURN_OBSERVATION_NAME,
                    as_type="agent",
                    input={"query": run.original_query},
                )
            )
            yield _Handle(span, CallbackHandler())

    def trace_turn(self, run: AgentRun) -> AbstractContextManager[TraceHandle]:
        return self._trace(run)

    def record_feedback(self, *, run_id: str, rating: str, note: str) -> None:
        self._client.create_score(
            name=FEEDBACK_SCORE_NAME,
            value=1.0 if rating == "up" else 0.0,
            trace_id=self.trace_id_for(run_id),
            comment=note or None,
            data_type="NUMERIC",
        )

    def flush(self) -> None:
        self._client.flush()

    def shutdown(self) -> None:
        try:
            self._client.shutdown()
        except Exception:
            logger.exception("langfuse shutdown failed")
