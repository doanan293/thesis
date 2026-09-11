"""System skills shipped in the repo: backend/skills/<skill-id>/SKILL.md."""

from collections.abc import Sequence
from pathlib import Path

from pharma_agent.domain.skill.models import Skill, SkillMetadata
from pharma_agent.domain.skill.parser import parse_skill_markdown


def load_system_skills(root: Path) -> list[Skill]:
    skills: list[Skill] = []
    if not root.exists():
        return skills
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        file = folder / "SKILL.md"
        if not file.exists():
            continue
        parsed = parse_skill_markdown(file.read_text(encoding="utf-8"))
        skills.append(
            Skill(
                skill_id=folder.name,
                owner_user_id=None,
                name=parsed.name,
                description=parsed.description,
                search_guidance=parsed.search_guidance,
                answer_guidance=parsed.answer_guidance,
                version=parsed.version,
                enabled=True,
            )
        )
    return skills


class FileSystemSkillCatalog:
    def __init__(self, root: Path) -> None:
        self._skills = load_system_skills(root)

    async def list_catalog(
        self, user_id: str | None, limit: int
    ) -> list[SkillMetadata]:
        return [s.metadata() for s in self._skills if s.enabled][:limit]

    async def get_by_ids(self, skill_ids: Sequence[str]) -> list[Skill]:
        wanted = set(skill_ids)
        return [s for s in self._skills if s.skill_id in wanted]
