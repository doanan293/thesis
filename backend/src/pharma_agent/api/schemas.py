from typing import Literal

from pydantic import BaseModel, Field

CONVERSATION_ID_PATTERN = r"^[0-9a-f]{32}$"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = Field(default=None, pattern=CONVERSATION_ID_PATTERN)


class RenameConversationRequest(BaseModel):
    title: str = Field(min_length=1, max_length=80)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    agent: bool
    checks: dict[str, bool]


class ErrorResponse(BaseModel):
    code: str
    message: str
