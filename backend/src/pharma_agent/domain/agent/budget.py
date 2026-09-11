from pydantic import BaseModel, ConfigDict

from pharma_agent.domain.llm.models import LlmUsage
from pharma_agent.domain.shared.errors import DomainError


class BudgetExhausted(DomainError):
    code = "BUDGET_EXHAUSTED"


class BudgetLimits(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_search_rounds: int = 3
    max_llm_calls: int = 10
    max_tokens: int = 40_000
    max_evidence_chars: int = 24_000
    deadline_seconds: float = 90.0


class BudgetUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    search_rounds: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def charged(self, usage: LlmUsage) -> "BudgetUsage":
        return self.model_copy(
            update={
                "llm_calls": self.llm_calls + 1,
                "prompt_tokens": self.prompt_tokens + usage.prompt_tokens,
                "completion_tokens": self.completion_tokens + usage.completion_tokens,
            }
        )

    def with_search(self) -> "BudgetUsage":
        return self.model_copy(update={"search_rounds": self.search_rounds + 1})
