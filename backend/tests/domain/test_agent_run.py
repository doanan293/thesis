import pytest

from pharma_agent.domain.agent.budget import BudgetExhausted
from pharma_agent.domain.agent.run import (
    AnswerMode,
    AnswerPlan,
    ErrorCode,
    EvidenceRequired,
    InvalidTransition,
    RunStatus,
    Step,
)
from pharma_agent.domain.agent.schemas import (
    Audience,
    Intent,
    JudgeOutcome,
    Language,
    RephraseResult,
)
from pharma_agent.domain.guardrail.models import Verdict, VerdictSource
from pharma_agent.domain.llm.models import LlmUsage
from pharma_agent.domain.retrieval.models import Query, QueryOrigin
from tests.domain.factories import NOW, make_run, search_result

PASS = Verdict(passed=True, in_scope=True, source=VerdictSource.LLM)
BLOCK = Verdict(
    passed=False, in_scope=False, source=VerdictSource.REGEX, label="jailbreak"
)
OFF_TOPIC = Verdict(
    passed=True, in_scope=False, source=VerdictSource.LLM, label="out_of_scope"
)
Q1 = [Query(text="paracetamol liều người lớn", origin=QueryOrigin.INITIAL)]


def rephrased(intent: Intent = Intent.PHARMA_QUESTION) -> RephraseResult:
    return RephraseResult(
        standalone_query="Liều paracetamol cho người lớn",
        audience=Audience.GENERAL_PUBLIC,
        language=Language.VI,
        intent=intent,
    )


def test_happy_path_search_judge_answer() -> None:
    run = make_run()
    run.record_guard(PASS, now=NOW)
    run.record_rephrase(rephrased(), now=NOW)
    assert run.standalone_query == "Liều paracetamol cho người lớn"
    assert run.allowed_steps() == {Step.SEARCH}

    run.record_search(Q1, search_result("c1", "c2"), now=NOW)
    assert run.allowed_steps() == {Step.JUDGE}
    assert run.usage.search_rounds == 1

    run.record_judge(JudgeOutcome.ANSWER, gaps=[], reason="đủ", now=NOW)
    assert run.allowed_steps() == {Step.ANSWER}

    plan = run.decide_answer()
    assert plan == AnswerPlan(mode=AnswerMode.GROUNDED, partial=False)
    run.submit_plan(plan, now=NOW)
    run.complete()
    assert run.status is RunStatus.COMPLETED
    assert run.allowed_steps() == frozenset()


def test_search_more_goes_through_refine_and_rejects_duplicate_queries() -> None:
    run = make_run()
    run.record_guard(PASS, now=NOW)
    run.record_rephrase(rephrased(), now=NOW)
    run.record_search(Q1, search_result("c1"), now=NOW)
    run.record_judge(
        JudgeOutcome.SEARCH_MORE, gaps=["liều tối đa"], reason="thiếu", now=NOW
    )
    assert run.allowed_steps() == {Step.REFINE}

    fresh = run.record_refine(
        ["Paracetamol LIỀU người lớn", "paracetamol liều tối đa mỗi ngày", ""], now=NOW
    )
    assert [q.text for q in fresh] == ["paracetamol liều tối đa mỗi ngày"]
    assert fresh[0].origin is QueryOrigin.REFINED
    assert run.allowed_steps() == {Step.SEARCH}

    run.record_search(fresh, search_result("c1", "c3"), now=NOW)
    run.record_judge(JudgeOutcome.SEARCH_MORE, gaps=["trẻ em"], reason="", now=NOW)
    only_dupes = run.record_refine(["paracetamol liều tối đa mỗi ngày"], now=NOW)
    assert only_dupes == []
    assert run.allowed_steps() == {Step.ANSWER}
    assert run.decide_answer() == AnswerPlan(mode=AnswerMode.GROUNDED, partial=True)


def test_search_round_limit_forces_partial_answer() -> None:
    run = make_run(max_search_rounds=1)
    run.record_guard(PASS, now=NOW)
    run.record_rephrase(rephrased(), now=NOW)
    run.record_search(Q1, search_result("c1"), now=NOW)
    run.record_judge(JudgeOutcome.SEARCH_MORE, gaps=["x"], reason="", now=NOW)
    assert run.allowed_steps() == {Step.ANSWER}
    assert run.decide_answer().partial is True


def test_no_evidence_leads_to_abstain() -> None:
    run = make_run()
    run.record_guard(PASS, now=NOW)
    run.record_rephrase(rephrased(), now=NOW)
    run.record_search(Q1, search_result(error="qdrant down"), now=NOW)
    run.record_judge(JudgeOutcome.ANSWER, gaps=[], reason="", now=NOW)
    assert run.decide_answer() == AnswerPlan(mode=AnswerMode.ABSTAIN)
    with pytest.raises(EvidenceRequired):
        run.submit_plan(AnswerPlan(mode=AnswerMode.GROUNDED), now=NOW)
    run.submit_plan(run.decide_answer(), now=NOW)
    run.complete()
    assert run.status is RunStatus.ABSTAINED


def test_guard_and_intent_short_circuit_to_answer() -> None:
    blocked = make_run()
    blocked.record_guard(BLOCK, now=NOW)
    assert blocked.allowed_steps() == {Step.ANSWER}
    assert blocked.decide_answer().mode is AnswerMode.BLOCKED

    off = make_run()
    off.record_guard(OFF_TOPIC, now=NOW)
    assert off.decide_answer().mode is AnswerMode.REDIRECT

    hello = make_run("xin chào")
    hello.record_guard(PASS, now=NOW)
    hello.record_rephrase(rephrased(Intent.SMALLTALK), now=NOW)
    assert hello.allowed_steps() == {Step.ANSWER}
    assert hello.decide_answer().mode is AnswerMode.NO_RETRIEVAL


def test_budget_reservation_and_exhaustion() -> None:
    run = make_run(max_llm_calls=4)
    assert run.can_afford_rephrase() is True
    run.charge(LlmUsage(prompt_tokens=10, completion_tokens=1))
    run.charge(LlmUsage(prompt_tokens=10, completion_tokens=1))
    assert run.usage.llm_calls == 2
    assert run.can_afford_rephrase() is False  # 2 + 3 > 4
    run.charge(LlmUsage())
    run.charge(LlmUsage())
    with pytest.raises(BudgetExhausted):
        run.charge(LlmUsage())
    assert run.usage.llm_calls == 5

    tokens = make_run(max_tokens=100)
    with pytest.raises(BudgetExhausted):
        tokens.charge(LlmUsage(prompt_tokens=90, completion_tokens=20))


def test_judge_failure_marks_partial_and_terminal_transitions_are_guarded() -> None:
    run = make_run()
    run.record_guard(PASS, now=NOW)
    run.record_rephrase(None, failed=True, now=NOW)
    assert run.standalone_query == run.original_query
    run.record_search(Q1, search_result("c1"), now=NOW)
    run.record_judge(
        JudgeOutcome.ANSWER, gaps=[], reason="judge failed", failed=True, now=NOW
    )
    assert run.decide_answer().partial is True

    run.fail(ErrorCode.ANSWER_FAILED, detail="boom")
    assert run.status is RunStatus.ERROR and run.error_code is ErrorCode.ANSWER_FAILED
    with pytest.raises(InvalidTransition):
        run.timeout()
    trace = run.to_trace()
    assert trace["status"] == "error" and trace["usage"]["search_rounds"] == 1
    assert [a["kind"] for a in trace["actions"]] == [
        "guard",
        "rephrase",
        "search",
        "judge",
    ]
