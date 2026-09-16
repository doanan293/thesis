"""Run one golden item through the backend chat graph (spec §6)."""

from __future__ import annotations

import time

from pharma_agent.application.chat.context import PipelineOptions, TurnDeps
from pharma_agent.application.chat.graph import ChatGraph
from pharma_agent.application.chat.runner import ChatTurnRunner, TurnOutcome
from pharma_agent.domain.agent.actions import ActionKind
from pharma_agent.domain.agent.budget import BudgetLimits
from pharma_agent.domain.agent.run import RunStatus
from pharma_agent.domain.conversation.models import ConversationContext, Turn
from pharma_agent.domain.guardrail.service import GuardrailService
from pharma_agent.domain.llm.port import LlmPort
from pharma_agent.domain.retrieval.service import RetrievalService
from pharma_agent.domain.shared.clock import Clock, SystemClock

from pharma_lab.bundle.chunks import gold_chunk_label
from pharma_lab.e2e.golden import GoldenItem
from pharma_lab.e2e.recording_llm import RecordingLlm
from pharma_lab.e2e.records import AnswerRecord, CitedSection, TokenUsage

E2E_USER = "e2e-evaluation"


def conversation_for(item: GoldenItem) -> ConversationContext:
    return ConversationContext(
        turns=[
            Turn(user_text=user, assistant_text=assistant, status="completed")
            for user, assistant in item.history
        ]
    )


def answer_record(
    item: GoldenItem,
    config: str,
    outcome: TurnOutcome,
    usage: dict[str, TokenUsage],
    latency: float,
) -> AnswerRecord:
    run = outcome.run
    hits = {
        evidence.hit.chunk_version_id: evidence.hit
        for evidence in run.evidence.active()
    }
    citations = []
    for citation in outcome.citations:
        hit = hits.get(citation.chunk_version_id)
        citations.append(
            CitedSection(
                index=citation.index,
                chunk_version_id=str(citation.chunk_version_id),
                chunk_id=None
                if hit is None
                else gold_chunk_label(hit.section_key, hit.ordinal),
                section_id=None if hit is None else hit.section_key,
            )
        )
    entries = run.actions.entries
    failed = run.status in (RunStatus.ERROR, RunStatus.TIMEOUT)
    return AnswerRecord(
        item_id=item.item_id,
        config=config,
        status=run.status.value,
        answer_mode=None if run.plan is None else run.plan.mode.value,
        answer_text=outcome.answer_text,
        context_text=outcome.context_text,
        citations=citations,
        retrieved_chunk_ids=[
            gold_chunk_label(e.hit.section_key, e.hit.ordinal)
            for e in run.evidence.active()
        ],
        standalone_query=run.standalone_query,
        search_queries=[
            list(action.payload.get("queries", []))
            for action in entries
            if action.kind is ActionKind.SEARCH
        ],
        judge_outcomes=[
            action.outcome for action in entries if action.kind is ActionKind.JUDGE
        ],
        llm_calls=run.usage.llm_calls,
        usage_by_role=usage,
        latency_seconds=round(latency, 3),
        non_llm_seconds=round(
            max(0.0, latency - sum(entry.seconds for entry in usage.values())), 3
        ),
        error=(
            f"{run.error_code.value}: {run.error_detail}".strip()
            if run.error_code is not None
            else None
        ),
        retryable=failed,
    )


class BackendTurnExecutor:
    """Runs a golden item as one chat turn with its scripted history."""

    def __init__(
        self,
        *,
        graph: ChatGraph,
        llm: LlmPort,
        retrieval: RetrievalService,
        limits: BudgetLimits,
        pipeline: PipelineOptions,
        config: str,
        clock: Clock | None = None,
    ) -> None:
        self._graph = graph
        self._llm = llm
        self._retrieval = retrieval
        self._limits = limits
        self._pipeline = pipeline
        self._config = config
        self._clock = clock if clock is not None else SystemClock()

    async def execute(self, item: GoldenItem) -> AnswerRecord:
        recorder = RecordingLlm(self._llm)
        deps = TurnDeps(
            llm=recorder,
            guardrail=GuardrailService(recorder),
            retrieval=self._retrieval,
            clock=self._clock,
        )
        runner = ChatTurnRunner(
            self._graph, deps, self._limits, pipeline=self._pipeline
        )
        execution = runner.start(
            user_id=E2E_USER,
            message=item.question,
            conversation=conversation_for(item),
        )
        started = time.perf_counter()
        async for _ in execution.events():
            pass
        latency = time.perf_counter() - started
        if execution.outcome is None:
            raise RuntimeError(f"{item.item_id}: the chat turn produced no outcome")
        return answer_record(
            item, self._config, execution.outcome, recorder.usage_by_role(), latency
        )
