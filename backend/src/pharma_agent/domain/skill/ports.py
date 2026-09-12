from collections.abc import Sequence
from typing import Protocol

from pharma_agent.domain.skill.models import Skill, SkillMetadata


class SkillCatalog(Protocol):
    async def list_catalog(
        self, user_id: str | None, limit: int
    ) -> list[SkillMetadata]:
        """Enabled system skills, then the user's enabled skills, at most `limit`.

        When a user skill has the same name as a system skill, only the system one is listed.
        """
        ...

    async def get_by_names(
        self, user_id: str | None, names: Sequence[str]
    ) -> list[Skill]:
        """Enabled skills visible to the user with those names (system skills win)."""
        ...


class SkillRepository(SkillCatalog, Protocol):
    async def replace_system(self, skills: Sequence[Skill]) -> None:
        """Make the stored system skills exactly `skills`: insert, update, delete."""
        ...

    async def create(self, skill: Skill) -> None:
        """Store a user skill. Raises DuplicateSkillName if the owner already has the name."""
        ...

    async def list_system(self) -> list[Skill]:
        """All system skills, ordered by name."""
        ...

    async def list_for_user(self, user_id: str) -> list[Skill]:
        """All skills owned by the user, enabled or not, ordered by name."""
        ...

    async def set_enabled(
        self, user_id: str, name: str, enabled: bool
    ) -> Skill | None: ...

    async def delete_owned(self, user_id: str, name: str) -> bool: ...
