from pydantic import BaseModel, ConfigDict, Field

from pharma_agent.domain.agent.run import SelectedSkill

SKILL_ID_PATTERN = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"
MAX_CATALOG_SIZE = 30
MAX_SELECTED_SKILLS = 3


class SkillMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    skill_id: str
    name: str
    description: str


class Skill(BaseModel):
    skill_id: str = Field(pattern=SKILL_ID_PATTERN)
    owner_user_id: str | None = None
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    search_guidance: str = ""
    answer_guidance: str = ""
    version: str = Field(min_length=1)
    enabled: bool = True

    @property
    def is_system(self) -> bool:
        return self.owner_user_id is None

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            skill_id=self.skill_id, name=self.name, description=self.description
        )

    def to_selected(self) -> SelectedSkill:
        return SelectedSkill(
            skill_id=self.skill_id,
            name=self.name,
            search_guidance=self.search_guidance,
            answer_guidance=self.answer_guidance,
        )
