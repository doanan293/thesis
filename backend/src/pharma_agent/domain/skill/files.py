"""System skills shipped in the repo: <root>/<name>/SKILL.md."""

from pathlib import Path

from pharma_agent.domain.skill.models import Skill
from pharma_agent.domain.skill.parser import SkillParseError

SKILL_FILE_NAMES = ("SKILL.md", "skill.md")


def load_system_skills(root: Path) -> list[Skill]:
    """Load every skill folder, validated like `skills-ref validate <root>/<name>`.

    Invalid skills fail loudly with every problem listed, prefixed by folder name.
    """
    skills: list[Skill] = []
    problems: list[str] = []
    if not root.exists():
        return skills
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        file = next(
            (folder / name for name in SKILL_FILE_NAMES if (folder / name).exists()),
            None,
        )
        if file is None:
            continue
        try:
            skills.append(
                Skill.from_markdown(
                    file.read_text(encoding="utf-8"), directory_name=folder.name
                )
            )
        except SkillParseError as exc:
            problems.extend(f"{folder.name}: {error}" for error in exc.errors)
    if problems:
        raise SkillParseError(problems)
    return skills
