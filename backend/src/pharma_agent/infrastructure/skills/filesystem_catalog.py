"""System skills shipped in the repo: backend/skills/<name>/SKILL.md."""

from collections.abc import Sequence
from pathlib import Path

from pharma_agent.domain.skill.files import load_system_skills
from pharma_agent.domain.skill.models import Skill, SkillMetadata


class FileSystemSkillCatalog:
    def __init__(self, root: Path) -> None:
        self._skills = load_system_skills(root)

    async def list_catalog(
        self, user_id: str | None, limit: int
    ) -> list[SkillMetadata]:
        return [s.metadata() for s in self._skills if s.enabled][:limit]

    async def get_by_names(
        self, user_id: str | None, names: Sequence[str]
    ) -> list[Skill]:
        wanted = set(names)
        return [s for s in self._skills if s.enabled and s.name in wanted]
