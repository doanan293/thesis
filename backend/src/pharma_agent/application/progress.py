"""Public progress contract emitted during a chat turn. Never expose node names, prompts or reasoning."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Phase(StrEnum):
    GUARDING = "guarding"
    UNDERSTANDING = "understanding"
    SELECTING_SKILLS = "selecting_skills"
    SEARCHING = "searching"
    READING = "reading"
    ANSWERING = "answering"
    DONE = "done"


class EventType(StrEnum):
    CONVERSATION = "conversation"
    PHASE = "phase"
    SKILLS_SELECTED = "skills_selected"
    EVIDENCE = "evidence"
    TOKEN = "token"
    CITATIONS = "citations"
    DONE = "done"
    ERROR = "error"


class ProgressEvent(BaseModel):
    type: EventType
    data: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def phase(cls, phase: Phase, **extra: Any) -> "ProgressEvent":
        return cls(type=EventType.PHASE, data={"phase": phase.value, **extra})

    @classmethod
    def token(cls, text: str) -> "ProgressEvent":
        return cls(type=EventType.TOKEN, data={"text": text})

    @classmethod
    def error(cls, code: str, message: str) -> "ProgressEvent":
        return cls(type=EventType.ERROR, data={"code": code, "message": message})
