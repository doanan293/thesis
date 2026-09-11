from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int
    chunk_id: str
    section_id: str
    title: str
    section: str
    start_page: int
    end_page: int
    table_id: str = ""


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
