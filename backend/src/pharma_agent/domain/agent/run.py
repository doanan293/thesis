from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from pharma_agent.domain.agent.actions import Action, ActionKind, ActionLog
from pharma_agent.domain.agent.budget import BudgetExhausted, BudgetLimits, BudgetUsage
from pharma_agent.domain.agent.schemas import (
    Audience,
    Intent,
    JudgeOutcome,
    Language,
    RephraseResult,
)
from pharma_agent.domain.guardrail.models import Verdict
from pharma_agent.domain.llm.models import LlmUsage
from pharma_agent.domain.retrieval.evidence import EvidenceSet
from pharma_agent.domain.retrieval.models import Query, QueryOrigin
from pharma_agent.domain.retrieval.service import SearchResult
from pharma_agent.domain.shared.errors import DomainError
from pharma_agent.domain.shared.ids import new_id

RESERVED_CALLS = 3  # judge + answer + one spare, kept free for rephrase


class EvidenceRequired(DomainError):
    code = "EVIDENCE_REQUIRED"


class InvalidTransition(DomainError):
    code = "INVALID_TRANSITION"


class Step(StrEnum):
    SEARCH = "search"
    JUDGE = "judge"
    REFINE = "refine"
    ANSWER = "answer"


class AnswerMode(StrEnum):
    GROUNDED = "grounded"
    NO_RETRIEVAL = "no_retrieval"
    ABSTAIN = "abstain"
    BLOCKED = "blocked"
    REDIRECT = "redirect"


class AnswerPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: AnswerMode
    partial: bool = False


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    ABSTAINED = "abstained"
    BLOCKED = "blocked"
    REDIRECTED = "redirected"
    ERROR = "error"
    TIMEOUT = "timeout"


class ErrorCode(StrEnum):
    GUARDRAIL_LLM_FAILED = "GUARDRAIL_LLM_FAILED"
    REPHRASE_FAILED = "REPHRASE_FAILED"
    SEARCH_FAILED = "SEARCH_FAILED"
    RERANK_FAILED = "RERANK_FAILED"
    JUDGE_FAILED = "JUDGE_FAILED"
    REFINE_FAILED = "REFINE_FAILED"
    ANSWER_FAILED = "ANSWER_FAILED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    EVIDENCE_REQUIRED = "EVIDENCE_REQUIRED"
    INTERNAL = "INTERNAL"


_FINAL_STATUS = {
    AnswerMode.GROUNDED: RunStatus.COMPLETED,
    AnswerMode.NO_RETRIEVAL: RunStatus.COMPLETED,
    AnswerMode.ABSTAIN: RunStatus.ABSTAINED,
    AnswerMode.BLOCKED: RunStatus.BLOCKED,
    AnswerMode.REDIRECT: RunStatus.REDIRECTED,
}


class AgentRun(BaseModel):
    """One chat turn. Owns the budget, the evidence, the action log and the loop rules."""

    run_id: str
    user_id: str
    conversation_id: str | None = None
    original_query: str
    standalone_query: str
    audience: Audience = Audience.UNKNOWN
    language: Language = Language.VI
    intent: Intent = Intent.PHARMA_QUESTION
    limits: BudgetLimits
    usage: BudgetUsage = Field(default_factory=BudgetUsage)
    evidence: EvidenceSet = Field(default_factory=EvidenceSet)
    actions: ActionLog = Field(default_factory=ActionLog)
    plan: AnswerPlan | None = None
    status: RunStatus = RunStatus.RUNNING
    error_code: ErrorCode | None = None
    error_detail: str = ""
    guard_verdict: Verdict | None = None
    used_queries: list[str] = Field(default_factory=list)
    last_judge: JudgeOutcome | None = None
    partial_reason: str | None = None
    started_at: datetime

    @classmethod
    def start(
        cls,
        *,
        user_id: str,
        original_query: str,
        limits: BudgetLimits,
        now: datetime,
        conversation_id: str | None = None,
        run_id: str | None = None,
    ) -> "AgentRun":
        query = original_query.strip()
        return cls(
            run_id=run_id or new_id(),
            user_id=user_id,
            conversation_id=conversation_id,
            original_query=query,
            standalone_query=query,
            limits=limits,
            started_at=now,
        )

    # ----- state queries -------------------------------------------------

    @property
    def is_finished(self) -> bool:
        return self.status is not RunStatus.RUNNING

    @property
    def calls_left(self) -> int:
        return self.limits.max_llm_calls - self.usage.llm_calls

    def has_evidence(self) -> bool:
        return bool(self.evidence.active())

    def allowed_steps(self) -> frozenset[Step]:
        if self.is_finished:
            return frozenset()
        if self.guard_verdict is not None and not self.guard_verdict.allows_processing:
            return frozenset({Step.ANSWER})
        if self.intent is not Intent.PHARMA_QUESTION:
            return frozenset({Step.ANSWER})
        last = self.actions.last_of(
            ActionKind.SEARCH, ActionKind.JUDGE, ActionKind.REFINE
        )
        if last is None:
            return frozenset({Step.SEARCH})
        if last.kind is ActionKind.SEARCH:
            return frozenset({Step.JUDGE})
        if last.kind is ActionKind.JUDGE:
            if self.last_judge is JudgeOutcome.ANSWER:
                return frozenset({Step.ANSWER})
            can_refine = (
                self.usage.search_rounds < self.limits.max_search_rounds
                and self.calls_left >= 2
            )
            return frozenset({Step.REFINE}) if can_refine else frozenset({Step.ANSWER})
        return (
            frozenset({Step.SEARCH})
            if last.outcome == "ok"
            else frozenset({Step.ANSWER})
        )

    def can_afford_rephrase(self) -> bool:
        return self.usage.llm_calls + RESERVED_CALLS <= self.limits.max_llm_calls

    def charge(self, usage: LlmUsage) -> None:
        self.usage = self.usage.charged(usage)
        if self.usage.llm_calls > self.limits.max_llm_calls:
            raise BudgetExhausted(
                f"llm calls {self.usage.llm_calls} > {self.limits.max_llm_calls}"
            )
        if self.usage.total_tokens > self.limits.max_tokens:
            raise BudgetExhausted(
                f"tokens {self.usage.total_tokens} > {self.limits.max_tokens}"
            )

    # ----- recording -----------------------------------------------------

    def record_guard(
        self, verdict: Verdict, *, now: datetime, llm_failed: bool = False
    ) -> None:
        self.guard_verdict = verdict
        self._log(
            ActionKind.GUARD,
            now,
            outcome="pass"
            if verdict.allows_processing
            else ("blocked" if not verdict.passed else "out_of_scope"),
            reason=verdict.reason,
            payload={
                "source": verdict.source.value,
                "label": verdict.label,
                "llm_failed": llm_failed,
            },
        )

    def record_rephrase(
        self,
        result: RephraseResult | None,
        *,
        now: datetime,
        skipped: bool = False,
        failed: bool = False,
    ) -> None:
        if result is not None:
            self.standalone_query = (
                result.standalone_query.strip() or self.original_query
            )
            self.audience = result.audience
            self.language = result.language
            self.intent = result.intent
            outcome = "ok"
        else:
            outcome = "skipped" if skipped else "failed"
        self._log(
            ActionKind.REPHRASE,
            now,
            outcome=outcome,
            payload={
                "standalone_query": self.standalone_query,
                "intent": self.intent.value,
                "audience": self.audience.value,
            },
        )

    def record_search(
        self, queries: Sequence[Query], result: SearchResult, *, now: datetime
    ) -> None:
        self.evidence.supersede_all()
        active = self.evidence.merge(result.items)
        self.usage = self.usage.with_search()
        for query in queries:
            if query.normalized not in self.used_queries:
                self.used_queries.append(query.normalized)
        self._log(
            ActionKind.SEARCH,
            now,
            outcome="ok" if result.succeeded else "error",
            reason=result.error or "",
            payload={
                "queries": [q.text for q in queries],
                "hits": len(result.items),
                "active_evidence": len(active),
                "rerank_failed": result.rerank_failed,
                "rerank_scored": result.rerank_scored,
                "rerank_reused": result.rerank_reused,
            },
        )

    def record_judge(
        self,
        decision: JudgeOutcome,
        *,
        gaps: Sequence[str],
        reason: str,
        now: datetime,
        failed: bool = False,
    ) -> None:
        self.last_judge = decision
        if failed:
            self.partial_reason = "judge_failed"
        self._log(
            ActionKind.JUDGE,
            now,
            outcome="failed" if failed else decision.value,
            reason=reason,
            payload={"gaps": list(gaps)},
        )

    def skip_judge(self, *, now: datetime) -> None:
        """Answer from the evidence found so far without asking the judge."""
        self.last_judge = JudgeOutcome.ANSWER
        self._log(ActionKind.JUDGE, now, outcome="skipped")

    def record_refine(
        self, queries: Sequence[str], *, now: datetime, failed: bool = False
    ) -> list[Query]:
        fresh: list[Query] = []
        seen = set(self.used_queries)
        for text in queries:
            query = Query(text=text.strip(), origin=QueryOrigin.REFINED)
            if query.text and query.normalized not in seen:
                seen.add(query.normalized)
                fresh.append(query)
        outcome = "failed" if failed else ("ok" if fresh else "duplicate")
        if outcome != "ok":
            self.partial_reason = f"refine_{outcome}"
        self._log(
            ActionKind.REFINE,
            now,
            outcome=outcome,
            payload={"proposed": list(queries), "accepted": [q.text for q in fresh]},
        )
        return fresh

    # ----- answering -----------------------------------------------------

    def decide_answer(self) -> AnswerPlan:
        if self.guard_verdict is not None and not self.guard_verdict.passed:
            return AnswerPlan(mode=AnswerMode.BLOCKED)
        if self.guard_verdict is not None and not self.guard_verdict.in_scope:
            return AnswerPlan(mode=AnswerMode.REDIRECT)
        if self.intent is not Intent.PHARMA_QUESTION:
            return AnswerPlan(mode=AnswerMode.NO_RETRIEVAL)
        if not self.has_evidence():
            return AnswerPlan(mode=AnswerMode.ABSTAIN)
        forced = (
            self.last_judge is not JudgeOutcome.ANSWER
            or self.partial_reason is not None
        )
        return AnswerPlan(mode=AnswerMode.GROUNDED, partial=forced)

    def submit_plan(self, plan: AnswerPlan, *, now: datetime) -> None:
        if plan.mode is AnswerMode.GROUNDED and not self.has_evidence():
            raise EvidenceRequired("grounded answer requires active evidence")
        self.plan = plan
        self._log(
            ActionKind.ANSWER,
            now,
            outcome=plan.mode.value,
            payload={"partial": plan.partial},
        )

    def complete(self) -> None:
        if self.plan is None:
            raise InvalidTransition("complete() requires a submitted plan")
        self._require_running()
        if self.plan.mode is AnswerMode.GROUNDED and self.plan.partial:
            self.status = RunStatus.PARTIAL
        else:
            self.status = _FINAL_STATUS[self.plan.mode]

    def fail(self, code: ErrorCode, *, detail: str = "") -> None:
        self._require_running()
        self.status = RunStatus.ERROR
        self.error_code = code
        self.error_detail = detail[:500]

    def timeout(self) -> None:
        self._require_running()
        self.status = RunStatus.TIMEOUT
        self.error_code = ErrorCode.DEADLINE_EXCEEDED

    def to_trace(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "error_code": self.error_code.value if self.error_code else None,
            "usage": self.usage.model_dump(),
            "audience": self.audience.value,
            "intent": self.intent.value,
            "standalone_query": self.standalone_query,
            "actions": [a.model_dump(mode="json") for a in self.actions.entries],
        }

    # ----- internals -----------------------------------------------------

    def _require_running(self) -> None:
        if self.is_finished:
            raise InvalidTransition(
                f"run already finished with status {self.status.value}"
            )

    def _log(
        self,
        kind: ActionKind,
        now: datetime,
        *,
        outcome: str,
        reason: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.actions.append(
            Action(
                kind=kind, at=now, outcome=outcome, reason=reason, payload=payload or {}
            )
        )
