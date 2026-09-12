from pathlib import Path

from pydantic import BaseModel

from pharma_agent.application.errors import (
    ApplicationError,
    InvalidInput,
    PayloadTooLarge,
)
from pharma_agent.domain.skill.files import SKILL_FILE_NAMES, load_system_skills
from pharma_agent.domain.skill.models import DuplicateSkillName, Skill
from pharma_agent.domain.skill.parser import SkillParseError
from pharma_agent.domain.skill.ports import SkillRepository

MAX_SKILL_BYTES = 64 * 1024


class SkillNotFound(ApplicationError):
    code = "SKILL_NOT_FOUND"


class SkillNameTaken(ApplicationError):
    code = "SKILL_NAME_TAKEN"


class SkillView(BaseModel):
    name: str
    title: str
    description: str
    enabled: bool
    is_system: bool
    version: str

    @classmethod
    def of(cls, skill: Skill) -> "SkillView":
        return cls(
            name=skill.name,
            title=skill.title,
            description=skill.description,
            enabled=skill.enabled,
            is_system=skill.is_system,
            version=skill.version,
        )


class SkillService:
    def __init__(self, skills: SkillRepository) -> None:
        self._skills = skills

    async def sync_system(self, root: Path) -> int:
        skills = load_system_skills(root)
        await self._skills.replace_system(skills)
        return len(skills)

    async def list_for_user(self, user_id: str) -> list[SkillView]:
        return [SkillView.of(s) for s in await self._skills.list_for_user(user_id)]

    async def list_visible(self, user_id: str) -> list[SkillView]:
        """System skills first, then the user's own skills (enabled or not)."""
        system = await self._skills.list_system()
        own = await self._skills.list_for_user(user_id)
        return [SkillView.of(skill) for skill in (*system, *own)]

    async def upload(self, user_id: str, filename: str, content: bytes) -> SkillView:
        if Path(filename).name not in SKILL_FILE_NAMES:
            raise InvalidInput("the uploaded file must be named SKILL.md")
        if len(content) > MAX_SKILL_BYTES:
            raise PayloadTooLarge(
                f"SKILL.md must be at most {MAX_SKILL_BYTES // 1024} KB"
            )
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InvalidInput("SKILL.md must be UTF-8 encoded") from exc
        try:
            skill = Skill.from_markdown(text, owner_user_id=user_id)
        except SkillParseError as exc:
            raise InvalidInput(str(exc)) from exc
        if any(
            system.name == skill.name for system in await self._skills.list_system()
        ):
            raise SkillNameTaken(f"'{skill.name}' is already used by a system skill")
        try:
            await self._skills.create(skill)
        except DuplicateSkillName as exc:
            raise SkillNameTaken(
                f"you already have a skill named '{skill.name}'"
            ) from exc
        return SkillView.of(skill)

    async def set_enabled(self, user_id: str, name: str, enabled: bool) -> SkillView:
        skill = await self._skills.set_enabled(user_id, name, enabled)
        if skill is None:
            raise SkillNotFound(name)
        return SkillView.of(skill)

    async def delete(self, user_id: str, name: str) -> None:
        if not await self._skills.delete_owned(user_id, name):
            raise SkillNotFound(name)
