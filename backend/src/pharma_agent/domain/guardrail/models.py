from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class VerdictSource(StrEnum):
    REGEX = "regex"
    LLM = "llm"
    SKIPPED = "skipped"


class Verdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    passed: bool
    in_scope: bool
    source: VerdictSource
    label: str = ""
    reason: str = ""

    @property
    def allows_processing(self) -> bool:
        return self.passed and self.in_scope


class LlmGuardVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_attack: bool
    in_scope: bool
    reason: str
