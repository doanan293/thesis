"""Runs one chat turn: the graph executes in its own task and publishes events to a queue.

Keeping the deadline inside the producer task means cancellation never lands in the consumer
(an SSE writer awaiting a slow client), and a consumer that disconnects cancels the graph.
"""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import suppress
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.types import StreamMode
from pydantic import BaseModel, Field

from pharma_agent.application.chat.context import (
    PipelineOptions,
    TurnContext,
    TurnDeps,
)
from pharma_agent.application.chat.graph import ChatGraph
from pharma_agent.application.chat.state import ChatTurnState
from pharma_agent.application.progress import EventType, ProgressEvent
from pharma_agent.application.tracing import NullTracing, TraceHandle, TurnTracer
from pharma_agent.domain.agent.budget import BudgetLimits
from pharma_agent.domain.agent.prompts import fallback_text
from pharma_agent.domain.agent.run import AgentRun, ErrorCode
from pharma_agent.domain.conversation.models import Citation, ConversationContext


class TurnOutcome(BaseModel):
    run: AgentRun
    answer_text: str
    citations: list[Citation] = Field(default_factory=list)
    context_text: str = ""


class ChatTurnExecution:
    """One turn in flight: iterate `events()`, then read `outcome`."""

    def __init__(
        self,
        graph: ChatGraph,
        deps: TurnDeps,
        limits: BudgetLimits,
        tracer: TurnTracer,
        *,
        user_id: str,
        message: str,
        conversation: ConversationContext,
        conversation_id: str | None,
        pipeline: PipelineOptions,
    ) -> None:
        self._graph = graph
        self._pipeline = pipeline
        self._deps = deps
        self._limits = limits
        self._tracer = tracer
        self._conversation = conversation
        self.run = AgentRun.start(
            user_id=user_id,
            original_query=message,
            limits=limits,
            now=deps.clock.now(),
            conversation_id=conversation_id,
        )
        self.outcome: TurnOutcome | None = None
        self.producer_task: asyncio.Task[None] | None = None

    async def events(self) -> AsyncGenerator[ProgressEvent]:
        queue: asyncio.Queue[ProgressEvent | None] = asyncio.Queue()
        self.producer_task = asyncio.create_task(self._produce(queue))
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
        finally:
            task = self.producer_task
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    async def _produce(self, queue: asyncio.Queue[ProgressEvent | None]) -> None:
        with self._tracer.trace_turn(self.run) as handle:
            await self._run_graph(queue, handle)
            if self.outcome is not None:
                handle.finish(
                    status=self.outcome.run.status.value,
                    output=self.outcome.answer_text,
                )

    async def _run_graph(
        self, queue: asyncio.Queue[ProgressEvent | None], handle: TraceHandle
    ) -> None:
        final: ChatTurnState | None = None
        config: RunnableConfig = {
            "configurable": {"thread_id": self.run.run_id},
            "callbacks": list(handle.callbacks),
        }
        modes: list[StreamMode] = ["custom", "values"]
        context = TurnContext(
            deps=self._deps, conversation=self._conversation, pipeline=self._pipeline
        )
        try:
            async with asyncio.timeout(self._limits.deadline_seconds):
                async for mode, chunk in self._graph.astream(
                    ChatTurnState(run=self.run),
                    config=config,
                    context=context,
                    stream_mode=modes,
                ):
                    if mode == "custom":
                        queue.put_nowait(ProgressEvent.model_validate(chunk))
                    else:
                        final = _as_state(chunk)
        except TimeoutError:
            run = final.run if final is not None else self.run
            if not run.is_finished:
                run.timeout()
            self._finish(queue, run, fallback_text(run.status), [], emit_text=True)
            return
        except Exception as exc:
            run = final.run if final is not None else self.run
            if not run.is_finished:
                run.fail(ErrorCode.INTERNAL, detail=f"{type(exc).__name__}: {exc}")
            self._finish(queue, run, fallback_text(run.status), [], emit_text=True)
            return

        if final is None:
            run = self.run
            if not run.is_finished:
                run.fail(ErrorCode.INTERNAL, detail="graph produced no final state")
            self._finish(queue, run, fallback_text(run.status), [], emit_text=True)
            return
        self._finish(
            queue,
            final.run,
            final.answer_text,
            final.citations,
            emit_text=False,
            context_text=final.context_text,
        )

    def _finish(
        self,
        queue: asyncio.Queue[ProgressEvent | None],
        run: AgentRun,
        answer_text: str,
        citations: list[Citation],
        *,
        emit_text: bool,
        context_text: str = "",
    ) -> None:
        self.outcome = TurnOutcome(
            run=run,
            answer_text=answer_text,
            citations=citations,
            context_text=context_text,
        )
        if emit_text:
            queue.put_nowait(ProgressEvent.token(answer_text))
        queue.put_nowait(_done_event(run))
        queue.put_nowait(None)


def _as_state(chunk: Any) -> ChatTurnState:
    return (
        chunk
        if isinstance(chunk, ChatTurnState)
        else ChatTurnState.model_validate(chunk)
    )


def _done_event(run: AgentRun) -> ProgressEvent:
    return ProgressEvent(
        type=EventType.DONE,
        data={
            "run_id": run.run_id,
            "conversation_id": run.conversation_id,
            "status": run.status.value,
            "error_code": run.error_code.value if run.error_code else None,
            "usage": run.usage.model_dump(),
        },
    )


class ChatTurnRunner:
    def __init__(
        self,
        graph: ChatGraph,
        deps: TurnDeps,
        limits: BudgetLimits,
        tracer: TurnTracer | None = None,
        pipeline: PipelineOptions | None = None,
    ) -> None:
        self._graph = graph
        self._deps = deps
        self._limits = limits
        self._tracer: TurnTracer = tracer if tracer is not None else NullTracing()
        self.pipeline = pipeline if pipeline is not None else PipelineOptions()

    def start(
        self,
        *,
        user_id: str,
        message: str,
        conversation: ConversationContext | None = None,
        conversation_id: str | None = None,
    ) -> ChatTurnExecution:
        return ChatTurnExecution(
            self._graph,
            self._deps,
            self._limits,
            self._tracer,
            user_id=user_id,
            message=message,
            conversation=conversation or ConversationContext(),
            conversation_id=conversation_id,
            pipeline=self.pipeline,
        )
