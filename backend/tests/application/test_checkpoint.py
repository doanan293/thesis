import warnings

from pharma_agent.application.chat.checkpoint import (
    checkpoint_serializer,
    checkpoint_types,
)
from pharma_agent.application.chat.state import ChatTurnState
from pharma_agent.domain.agent.run import AgentRun, RunStatus
from pharma_agent.domain.agent.schemas import JudgeOutcome
from pharma_agent.domain.guardrail.models import Verdict, VerdictSource
from pharma_agent.domain.retrieval.models import HydrateStrategy, Query, QueryOrigin
from tests.domain.factories import NOW, chunk_uuid, make_run, search_result


def test_checkpoint_types_cover_nested_models_and_enums() -> None:
    types = set(checkpoint_types(ChatTurnState))
    assert {AgentRun, Verdict, VerdictSource, HydrateStrategy, JudgeOutcome} <= types
    assert RunStatus in types and Query in types and QueryOrigin in types


def test_agent_run_round_trips_under_strict_allowlist() -> None:
    run = make_run()
    run.record_guard(
        Verdict(passed=True, in_scope=True, source=VerdictSource.LLM), now=NOW
    )
    run.record_search(
        [Query(text="q", origin=QueryOrigin.INITIAL)], search_result("c1"), now=NOW
    )
    run.record_judge(JudgeOutcome.ANSWER, gaps=[], reason="ok", now=NOW)
    serde = checkpoint_serializer()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        restored = serde.loads_typed(serde.dumps_typed(run))
    assert isinstance(restored, AgentRun)
    assert (
        restored.evidence.items[0].hit.hydrate_strategy is HydrateStrategy.CHUNK_WINDOW
    )
    assert restored.last_judge is JudgeOutcome.ANSWER
    assert restored.guard_verdict is not None
    assert restored.guard_verdict.source is VerdictSource.LLM
    assert restored.evidence.items[0].hit.chunk_version_id == chunk_uuid("c1")
    assert restored.evidence.items[0].chunks[0].section_revision_id == (
        run.evidence.items[0].chunks[0].section_revision_id
    )
