from pydantic import BaseModel, ConfigDict

from pharma_agent.domain.guardrail.groups import regex_screen
from pharma_agent.domain.guardrail.models import LlmGuardVerdict, Verdict, VerdictSource
from pharma_agent.domain.guardrail.prompts import guardrail_messages
from pharma_agent.domain.llm.models import LlmRole, LlmUsage
from pharma_agent.domain.llm.port import LlmError, LlmPort


class GuardrailOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    verdict: Verdict
    usage: LlmUsage = LlmUsage()
    llm_failed: bool = False


class GuardrailService:
    """Regex first, then a small LLM classifier. Any LLM failure fails open."""

    def __init__(self, llm: LlmPort) -> None:
        self._llm = llm

    async def check(self, query: str) -> GuardrailOutcome:
        regex_verdict = regex_screen(query)
        if regex_verdict is not None:
            return GuardrailOutcome(verdict=regex_verdict)
        try:
            result, usage = await self._llm.structured(
                LlmRole.GUARDRAIL, guardrail_messages(query), LlmGuardVerdict
            )
        except LlmError as exc:
            return GuardrailOutcome(
                verdict=Verdict(
                    passed=True,
                    in_scope=True,
                    source=VerdictSource.SKIPPED,
                    reason=f"guardrail llm failed: {exc}",
                ),
                llm_failed=True,
            )
        if result.is_attack:
            verdict = Verdict(
                passed=False,
                in_scope=False,
                source=VerdictSource.LLM,
                label="llm_attack",
                reason=result.reason,
            )
        else:
            verdict = Verdict(
                passed=True,
                in_scope=result.in_scope,
                source=VerdictSource.LLM,
                label="" if result.in_scope else "out_of_scope",
                reason=result.reason,
            )
        return GuardrailOutcome(verdict=verdict, usage=usage)
