from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from pharma_agent.domain.retrieval.models import HydrateStrategy
from pharma_agent.domain.shared.errors import DomainError
from pharma_agent.domain.shared.ids import new_id


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class Citation(BaseModel):
    """A cited evidence block: the chunk version the model read and the blocks around it."""

    model_config = ConfigDict(frozen=True)

    index: int
    chunk_version_id: UUID
    release_id: UUID
    strategy: HydrateStrategy
    block_chunk_version_ids: list[UUID]
    source: str
    title: str
    section: str
    start_page: int | None
    end_page: int | None
    snippet: str


class Turn(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_text: str
    assistant_text: str
    status: str

    def char_count(self) -> int:
        return len(self.user_text) + len(self.assistant_text)


class ConversationContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    summary: str = ""
    turns: list[Turn] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.summary and not self.turns


class ConversationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str


MAX_TITLE_CHARS = 80


class InvalidTitle(DomainError):
    code = "INVALID_TITLE"


def _clean_title(text: str) -> str:
    return " ".join(text.split())


class Conversation(BaseModel):
    conversation_id: str
    user_id: str
    title: str
    summary: str = ""
    turn_count: int = 0
    summarized_turns: int = 0
    created_at: datetime
    updated_at: datetime

    @classmethod
    def start(
        cls,
        *,
        user_id: str,
        first_message: str,
        now: datetime,
        conversation_id: str | None = None,
    ) -> "Conversation":
        title = (
            _clean_title(first_message)[:MAX_TITLE_CHARS].rstrip()
            or "Cuộc trò chuyện mới"
        )
        return cls(
            conversation_id=conversation_id or new_id(),
            user_id=user_id,
            title=title,
            created_at=now,
            updated_at=now,
        )

    def record_turn(self, now: datetime) -> None:
        self.turn_count += 1
        self.updated_at = now

    def needs_summary(self, every: int) -> bool:
        return self.turn_count - self.summarized_turns >= every

    def apply_summary(self, summary: str, *, covered_turns: int, now: datetime) -> None:
        if covered_turns > self.turn_count or covered_turns < self.summarized_turns:
            raise ValueError(
                f"covered_turns={covered_turns} must be between "
                f"{self.summarized_turns} and {self.turn_count}"
            )
        self.summary = summary.strip()
        self.summarized_turns = covered_turns
        self.updated_at = now

    def rename(self, title: str, *, now: datetime) -> None:
        cleaned = _clean_title(title)
        if not cleaned or len(cleaned) > MAX_TITLE_CHARS:
            raise InvalidTitle(f"title must be 1-{MAX_TITLE_CHARS} characters")
        self.title = cleaned
        self.updated_at = now


class Message(BaseModel):
    message_id: str
    conversation_id: str
    role: MessageRole
    content: str
    status: str
    citations: list[Citation] = Field(default_factory=list)
    phases: list[str] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    run_id: str | None = None
    created_at: datetime
