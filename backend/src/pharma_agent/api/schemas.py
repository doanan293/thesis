from typing import Literal

from pydantic import BaseModel, Field

CONVERSATION_ID_PATTERN = r"^[0-9a-f]{32}$"
MESSAGE_ID_PATTERN = CONVERSATION_ID_PATTERN


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = Field(default=None, pattern=CONVERSATION_ID_PATTERN)


class RenameConversationRequest(BaseModel):
    title: str = Field(min_length=1, max_length=80)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    agent: bool
    checks: dict[str, bool]
    # Reason code of each failed check that declares one, e.g. {"corpus": "CORPUS_NOT_READY"}.
    reasons: dict[str, str] = Field(default_factory=dict)


class ProblemItem(BaseModel):
    """One invalid request field of a 422 problem."""

    loc: list[str | int]
    message: str
    type: str


class Problem(BaseModel):
    """RFC 9457 problem details; clients switch on `code`."""

    type: str
    title: str
    status: int
    detail: str | None = None
    code: str
    errors: list[ProblemItem] | None = None


class FeedbackRequest(BaseModel):
    rating: Literal["up", "down"]
    note: str = Field(default="", max_length=2000)
