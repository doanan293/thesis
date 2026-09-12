from collections.abc import Sequence
from typing import Protocol

from pharma_agent.domain.skill.models import Skill, SkillMetadata


class SkillCatalog(Protocol):
    async def list_catalog(
        self, user_id: str | None, limit: int
    ) -> list[SkillMetadata]:
        """Enabled system skills plus the user's enabled skills, at most `limit` entries."""
        ...

    async def get_by_ids(self, skill_ids: Sequence[str]) -> list[Skill]: ...


class SkillRepository(SkillCatalog, Protocol):
    async def upsert_system(self, skills: Sequence[Skill]) -> None:
        """Insert or update system skills (owner None) by skill_id."""
        ...

    async def create(self, skill: Skill) -> None: ...

    async def list_for_user(self, user_id: str) -> list[Skill]:
        """All skills owned by the user, enabled or not, ordered by name."""
        ...

    async def get_owned(self, user_id: str, skill_id: str) -> Skill | None: ...

    async def set_enabled(
        self, user_id: str, skill_id: str, enabled: bool
    ) -> Skill | None: ...

    async def delete_owned(self, user_id: str, skill_id: str) -> bool: ...
