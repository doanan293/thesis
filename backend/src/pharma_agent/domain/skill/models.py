import re

from pydantic import BaseModel, ConfigDict, Field

from pharma_agent.domain.agent.run import SelectedSkill
from pharma_agent.domain.shared.errors import DomainError
from pharma_agent.domain.skill.parser import (
    MAX_DESCRIPTION_CHARS,
    MAX_NAME_CHARS,
    parse_skill_markdown,
)

MAX_CATALOG_SIZE = 30
MAX_SELECTED_SKILLS = 3
_TITLE_HEADING = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


class DuplicateSkillName(DomainError):
    code = "DUPLICATE_SKILL_NAME"


class SkillMetadata(BaseModel):
    """What the skill selector sees: the Agent Skills `name` and `description`."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str


class Skill(BaseModel):
    """A validated Agent Skill; `content` is the SKILL.md exactly as shipped or uploaded."""

    name: str = Field(min_length=1, max_length=MAX_NAME_CHARS)
    description: str = Field(min_length=1, max_length=MAX_DESCRIPTION_CHARS)
    instructions: str = ""
    content: str = Field(min_length=1)
    version: str = Field(min_length=1)
    owner_user_id: str | None = None
    enabled: bool = True

    @classmethod
    def from_markdown(
        cls,
        content: str,
        *,
        owner_user_id: str | None = None,
        directory_name: str | None = None,
        enabled: bool = True,
    ) -> "Skill":
        parsed = parse_skill_markdown(content, directory_name=directory_name)
        return cls(
            name=parsed.name,
            description=parsed.description,
            instructions=parsed.instructions,
            content=content,
            version=parsed.version,
            owner_user_id=owner_user_id,
            enabled=enabled,
        )

    @property
    def is_system(self) -> bool:
        return self.owner_user_id is None

    @property
    def title(self) -> str:
        """The first `# ` heading of the body, falling back to `name`."""
        match = _TITLE_HEADING.search(self.instructions)
        return match.group(1) if match else self.name

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(name=self.name, description=self.description)

    def to_selected(self) -> SelectedSkill:
        return SelectedSkill(
            name=self.name, title=self.title, instructions=self.instructions
        )
