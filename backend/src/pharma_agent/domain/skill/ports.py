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
