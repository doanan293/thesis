"""System skills shipped in the repo: <root>/<skill-id>/SKILL.md."""

from pathlib import Path

from pharma_agent.domain.skill.models import Skill
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
