"""Graph nodes: thin wrappers that call the domain and emit public progress events."""

import functools
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any, Protocol

from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from pharma_agent.application.chat.context import TurnContext
from pharma_agent.application.chat.state import ChatTurnState
from pharma_agent.application.progress import EventType, Phase, ProgressEvent
from pharma_agent.domain.agent.budget import BudgetExhausted
from pharma_agent.domain.agent.citations import CitationSanitizer, citations_from
from pharma_agent.domain.agent.prompts import (
    answer_messages,
    fallback_text,
    judge_messages,
    refine_messages,
    rephrase_messages,
    skill_selection_messages,
)
from pharma_agent.domain.agent.run import AnswerMode, ErrorCode, OptionalStep
from pharma_agent.domain.agent.schemas import (
    JudgeDecision,
    JudgeOutcome,
    RefineResult,
    RephraseResult,
    SkillSelection,
)
from pharma_agent.domain.guardrail.models import VerdictSource
from pharma_agent.domain.llm.models import LlmRole, LlmUsage
from pharma_agent.domain.llm.port import LlmError
from pharma_agent.domain.retrieval.models import Query, QueryOrigin
from pharma_agent.domain.skill.models import MAX_CATALOG_SIZE
from pharma_agent.domain.skill.resolver import resolve_selected

NodeUpdate = dict[str, Any]


class Node(Protocol):
    def __call__(
        self, state: ChatTurnState, *, runtime: Runtime[TurnContext]
    ) -> Awaitable[NodeUpdate]: ...


def _emit(event: ProgressEvent) -> None:
    get_stream_writer()(event.model_dump(mode="json"))


def guarded(code: ErrorCode) -> Callable[[Node], Node]:
    """Last line of defence: any unexpected exception fails the run so routing sends it to fallback."""

    def decorator(fn: Node) -> Node:
        @functools.wraps(fn)
        async def wrapper(
            state: ChatTurnState, *, runtime: Runtime[TurnContext]
        ) -> NodeUpdate:
            try:
                return await fn(state, runtime=runtime)
            except Exception as exc:
                if not state.run.is_finished:
                    state.run.fail(code, detail=f"{type(exc).__name__}: {exc}")
                return {"run": state.run}

        return wrapper

    return decorator


@guarded(ErrorCode.INTERNAL)
async def guard_node(
    state: ChatTurnState, *, runtime: Runtime[TurnContext]
) -> NodeUpdate:
    run, deps = state.run, runtime.context.deps
    _emit(ProgressEvent.phase(Phase.GUARDING))
    outcome = await deps.guardrail.check(run.original_query)
    if outcome.verdict.source is VerdictSource.LLM:
        with suppress(BudgetExhausted):
            run.charge(outcome.usage)
    run.record_guard(
        outcome.verdict, now=deps.clock.now(), llm_failed=outcome.llm_failed
    )
    return {"run": run}


@guarded(ErrorCode.REPHRASE_FAILED)
async def rephrase_node(
    state: ChatTurnState, *, runtime: Runtime[TurnContext]
) -> NodeUpdate:
    run, deps = state.run, runtime.context.deps
    _emit(ProgressEvent.phase(Phase.UNDERSTANDING))
    now = deps.clock.now()
    if not run.can_afford(OptionalStep.REPHRASE):
        run.record_rephrase(None, now=now, skipped=True)
        return {"run": run}
    try:
        result, usage = await deps.llm.structured(
            LlmRole.REPHRASE,
            rephrase_messages(run.original_query, runtime.context.conversation),
            RephraseResult,
        )
        run.charge(usage)
    except (LlmError, BudgetExhausted):
        run.record_rephrase(None, now=now, failed=True)
        return {"run": run}
    run.record_rephrase(result, now=now)
    return {"run": run}


@guarded(ErrorCode.SKILL_RESOLUTION_FAILED)
async def resolve_skills_node(
    state: ChatTurnState, *, runtime: Runtime[TurnContext]
) -> NodeUpdate:
    run, deps = state.run, runtime.context.deps
    _emit(ProgressEvent.phase(Phase.SELECTING_SKILLS))
    now = deps.clock.now()
    if not run.can_afford(OptionalStep.RESOLVE_SKILLS):
        run.record_skills([], now=now)
        return {"run": run}
    catalog = await deps.skills.list_catalog(run.user_id, limit=MAX_CATALOG_SIZE + 1)
    if not catalog or len(catalog) > MAX_CATALOG_SIZE:
        run.record_skills([], now=now, failed=len(catalog) > MAX_CATALOG_SIZE)
        return {"run": run}
    try:
        selection, usage = await deps.llm.structured(
            LlmRole.SKILL_SELECTOR,
            skill_selection_messages(run.standalone_query, catalog),
            SkillSelection,
        )
        run.charge(usage)
    except (LlmError, BudgetExhausted):
        run.record_skills([], now=now, failed=True)
        return {"run": run}
    names = resolve_selected(catalog, selection.skill_names)
    found = await deps.skills.get_by_names(run.user_id, names) if names else []
    by_name = {skill.name: skill for skill in found}
    selected = [by_name[name].to_selected() for name in names if name in by_name]
    run.record_skills(selected, now=now)
    if selected:
        _emit(
            ProgressEvent(
                type=EventType.SKILLS_SELECTED,
                data={"skills": [{"name": s.name, "title": s.title} for s in selected]},
            )
        )
    return {"run": run}


@guarded(ErrorCode.SEARCH_FAILED)
async def search_node(
    state: ChatTurnState, *, runtime: Runtime[TurnContext]
) -> NodeUpdate:
    run, deps = state.run, runtime.context.deps
    queries = state.pending_queries or [
        Query(text=run.standalone_query, origin=QueryOrigin.INITIAL)
    ]
    _emit(ProgressEvent.phase(Phase.SEARCHING, round=run.usage.search_rounds + 1))
    result = await deps.retrieval.search(
        queries,
        rerank_query=run.standalone_query,
        known_scores=run.evidence.rerank_scores(),
    )
    run.record_search(queries, result, now=deps.clock.now())
    if result.items:
        _emit(ProgressEvent.phase(Phase.READING))
    return {"run": run, "pending_queries": []}


@guarded(ErrorCode.JUDGE_FAILED)
async def judge_node(
    state: ChatTurnState, *, runtime: Runtime[TurnContext]
) -> NodeUpdate:
    run, deps = state.run, runtime.context.deps
    now = deps.clock.now()
    gaps: list[str] = []
    try:
        decision, usage = await deps.llm.structured(
            LlmRole.JUDGE,
            judge_messages(run, run.evidence.summary_view()),
            JudgeDecision,
        )
        run.charge(usage)
        run.record_judge(
            decision.decision, gaps=decision.gaps, reason=decision.reason, now=now
        )
        gaps = list(decision.gaps)
    except (LlmError, BudgetExhausted) as exc:
        run.record_judge(
            JudgeOutcome.ANSWER, gaps=[], reason=str(exc), now=now, failed=True
        )
    return {"run": run, "last_gaps": gaps}


@guarded(ErrorCode.REFINE_FAILED)
async def refine_node(
    state: ChatTurnState, *, runtime: Runtime[TurnContext]
) -> NodeUpdate:
    run, deps = state.run, runtime.context.deps
    now = deps.clock.now()
    hints: list[str] = []
    for evidence in run.evidence.active():
        hints.extend(h for h in evidence.hit.term_hints() if h not in hints)
    try:
        result, usage = await deps.llm.structured(
            LlmRole.REFINE,
            refine_messages(run, state.last_gaps, hints[:10]),
            RefineResult,
        )
        run.charge(usage)
        fresh = run.record_refine(result.queries, now=now)
    except (LlmError, BudgetExhausted):
        fresh = run.record_refine([], now=now, failed=True)
    return {"run": run, "pending_queries": fresh}


@guarded(ErrorCode.ANSWER_FAILED)
async def answer_node(
    state: ChatTurnState, *, runtime: Runtime[TurnContext]
) -> NodeUpdate:
    run, deps = state.run, runtime.context.deps
    _emit(ProgressEvent.phase(Phase.ANSWERING))
    plan = run.decide_answer()
    run.submit_plan(plan, now=deps.clock.now())

    packed = (
        run.evidence.pack(run.limits.max_evidence_chars)
        if plan.mode is AnswerMode.GROUNDED
        else []
    )
    context_text, numbered = run.evidence.context_view(packed)
    if numbered:
        _emit(
            ProgressEvent(
                type=EventType.EVIDENCE,
                data={
                    "items": [
                        {
                            "index": index,
                            "title": e.hit.title,
                            "section": e.hit.section,
                            "start_page": e.hit.start_page,
                            "end_page": e.hit.end_page,
                            "table_id": e.hit.table_id,
                        }
                        for index, e in numbered
                    ]
                },
            )
        )

    sanitizer = CitationSanitizer(index for index, _ in numbered)
    parts: list[str] = []
    usage = LlmUsage()
    async for delta in deps.llm.stream(
        LlmRole.ANSWER, answer_messages(run, plan, context_text)
    ):
        if delta.text:
            clean = sanitizer.feed(delta.text)
            if clean:
                parts.append(clean)
                _emit(ProgressEvent.token(clean))
        if delta.usage is not None:
            usage = delta.usage
    tail = sanitizer.flush()
    if tail:
        parts.append(tail)
        _emit(ProgressEvent.token(tail))
    with suppress(BudgetExhausted):
        run.charge(usage)

    citations = citations_from(numbered, sanitizer.used)
    _emit(
        ProgressEvent(
            type=EventType.CITATIONS,
            data={"items": [c.model_dump() for c in citations]},
        )
    )
    run.complete()
    return {"run": run, "answer_text": "".join(parts), "citations": citations}


async def fallback_node(
    state: ChatTurnState, *, runtime: Runtime[TurnContext]
) -> NodeUpdate:
    run = state.run
    if not run.is_finished:
        run.fail(ErrorCode.INTERNAL, detail="fallback reached without a recorded error")
    text = fallback_text(run.status)
    _emit(ProgressEvent.token(text))
    return {"run": run, "answer_text": text, "citations": []}
