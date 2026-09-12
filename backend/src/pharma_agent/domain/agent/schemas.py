"""Structured-output schemas returned by the LLM. No defaults: OpenAI strict mode requires every field."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, field_validator


class Audience(StrEnum):
    GENERAL_PUBLIC = "general_public"
    PROFESSIONAL = "professional"
    UNKNOWN = "unknown"


class Language(StrEnum):
    VI = "vi"
    EN = "en"
    OTHER = "other"


class Intent(StrEnum):
    PHARMA_QUESTION = "pharma_question"
    SMALLTALK = "smalltalk"
    META = "meta"


class JudgeOutcome(StrEnum):
    ANSWER = "answer"
    SEARCH_MORE = "search_more"


class RephraseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    standalone_query: str
    audience: Audience
    language: Language
    intent: Intent


class SkillSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill_names: list[str]


class JudgeDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: JudgeOutcome
    gaps: list[str]
    reason: str


class RefineResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    queries: list[str]

    @field_validator("queries")
    @classmethod
    def _clean(cls, value: list[str]) -> list[str]:
        cleaned = [q.strip() for q in value if q and q.strip()]
        return cleaned[:3]
