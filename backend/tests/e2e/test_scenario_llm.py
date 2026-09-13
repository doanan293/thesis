import asyncio

import pytest

from pharma_agent.domain.agent.schemas import (
    JudgeDecision,
    JudgeOutcome,
    RephraseResult,
)
from pharma_agent.domain.guardrail.models import LlmGuardVerdict
from pharma_agent.domain.guardrail.prompts import guardrail_messages
from pharma_agent.domain.llm.models import LlmRole
from tests.e2e.scenario_llm import GROUNDED_ANSWER, ScenarioLlm


async def test_default_question_passes_and_answers_with_a_citation() -> None:
    llm = ScenarioLlm()
    verdict, _ = await llm.structured(
        LlmRole.GUARDRAIL,
        guardrail_messages("Paracetamol người lớn uống bao nhiêu?"),
        LlmGuardVerdict,
    )
    assert verdict.is_attack is False and verdict.in_scope is True
    judge, _ = await llm.structured(LlmRole.JUDGE, [], JudgeDecision)
    assert judge.decision is JudgeOutcome.ANSWER
    text = "".join([delta.text or "" async for delta in llm.stream(LlmRole.ANSWER, [])])
    assert text == GROUNDED_ANSWER and "[1]" in text


async def test_blocked_marker_makes_the_guardrail_flag_an_attack() -> None:
    verdict, _ = await ScenarioLlm().structured(
        LlmRole.GUARDRAIL,
        guardrail_messages("[e2e:blocked] Paracetamol là gì?"),
        LlmGuardVerdict,
    )
    assert verdict.is_attack is True and verdict.in_scope is False


async def test_timeout_marker_hangs_the_guardrail_call() -> None:
    llm = ScenarioLlm(hang_seconds=60)
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.05):
            await llm.structured(
                LlmRole.GUARDRAIL,
                guardrail_messages("[e2e:timeout] Paracetamol là gì?"),
                LlmGuardVerdict,
            )


async def test_a_schema_the_scenario_does_not_script_is_a_test_bug() -> None:
    with pytest.raises(AssertionError, match="RephraseResult"):
        await ScenarioLlm().structured(
            LlmRole.GUARDRAIL, guardrail_messages("xin chào"), RephraseResult
        )
