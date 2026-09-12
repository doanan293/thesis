import secrets
from pathlib import Path

from pydantic import BaseModel

from pharma_agent.application.errors import ApplicationError, InvalidInput
from pharma_agent.domain.skill.files import load_system_skills
from pharma_agent.domain.skill.models import Skill
from pharma_agent.domain.skill.parser import SkillParseError, parse_skill_markdown
from pharma_agent.domain.skill.ports import SkillRepository
from pharma_agent.domain.skill.slug import slugify

MAX_SKILL_BYTES = 64 * 1024


class SkillNotFound(ApplicationError):
    code = "SKILL_NOT_FOUND"


class SkillView(BaseModel):
    id: str
    name: str
    description: str
    enabled: bool
    is_system: bool
    version: str

    @classmethod
    def of(cls, skill: Skill) -> "SkillView":
        return cls(
            id=skill.skill_id,
            name=skill.name,
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
        await self._skills.upsert_system(skills)
        return len(skills)

    async def list_for_user(self, user_id: str) -> list[SkillView]:
        return [SkillView.of(s) for s in await self._skills.list_for_user(user_id)]

    async def upload(self, user_id: str, filename: str, content: bytes) -> SkillView:
        if Path(filename).name.lower() != "skill.md":
            raise InvalidInput("the uploaded file must be named SKILL.md")
        if len(content) > MAX_SKILL_BYTES:
            raise InvalidInput(f"SKILL.md must be at most {MAX_SKILL_BYTES // 1024} KB")
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InvalidInput("SKILL.md must be UTF-8 encoded") from exc
        try:
            parsed = parse_skill_markdown(text)
        except SkillParseError as exc:
            raise InvalidInput(str(exc)) from exc
        skill = Skill(
            skill_id=f"{slugify(parsed.name)[:50]}-{secrets.token_hex(3)}",
            owner_user_id=user_id,
            name=parsed.name,
            description=parsed.description,
            search_guidance=parsed.search_guidance,
            answer_guidance=parsed.answer_guidance,
            version=parsed.version,
        )
        await self._skills.create(skill)
        return SkillView.of(skill)

    async def set_enabled(
        self, user_id: str, skill_id: str, enabled: bool
    ) -> SkillView:
        skill = await self._skills.set_enabled(user_id, skill_id, enabled)
        if skill is None:
            raise SkillNotFound(skill_id)
        return SkillView.of(skill)

    async def delete(self, user_id: str, skill_id: str) -> None:
        if not await self._skills.delete_owned(user_id, skill_id):
            raise SkillNotFound(skill_id)
