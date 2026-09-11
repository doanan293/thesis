from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ActionKind(StrEnum):
    GUARD = "guard"
    REPHRASE = "rephrase"
    RESOLVE_SKILLS = "resolve_skills"
    SEARCH = "search"
    JUDGE = "judge"
    REFINE = "refine"
    ANSWER = "answer"


class Action(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: ActionKind
    at: datetime
    outcome: str = ""
    reason: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


class ActionLog(BaseModel):
    entries: list[Action] = Field(default_factory=list)

    def append(self, action: Action) -> None:
        self.entries.append(action)

    def last_of(self, *kinds: ActionKind) -> Action | None:
        for action in reversed(self.entries):
            if action.kind in kinds:
                return action
        return None

    def count(self, kind: ActionKind) -> int:
        return sum(1 for a in self.entries if a.kind is kind)
