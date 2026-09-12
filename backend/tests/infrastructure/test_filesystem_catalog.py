from pathlib import Path

import pytest
from skills_ref.validator import validate

from pharma_agent.domain.agent.prompts import DISCLAIMER_PHRASES
from pharma_agent.domain.skill.parser import SkillParseError
from pharma_agent.infrastructure.skills.filesystem_catalog import (
    FileSystemSkillCatalog,
    load_system_skills,
)

SKILLS_DIR = Path(__file__).resolve().parents[2] / "skills"
EXPECTED_NAMES = {
    "brand-to-generic",
    "dosing-by-population",
    "drug-interaction",
    "drug-monograph",
    "plain-language",
}


@pytest.mark.parametrize(
    "folder",
    sorted(path for path in SKILLS_DIR.iterdir() if path.is_dir()),
    ids=lambda path: path.name,
)
def test_repo_skill_passes_the_official_validator(folder: Path) -> None:
    assert validate(folder) == []


def test_repo_skills_load_with_folder_names_and_instructions() -> None:
    skills = load_system_skills(SKILLS_DIR)
    assert {s.name for s in skills} == EXPECTED_NAMES
    for skill in skills:
        assert skill.is_system and skill.enabled and skill.instructions
        assert skill.title != skill.name, skill.name
        lowered = skill.instructions.lower()
        assert not any(phrase in lowered for phrase in DISCLAIMER_PHRASES), skill.name


async def test_catalog_lists_metadata_and_fetches_by_name() -> None:
    catalog = FileSystemSkillCatalog(SKILLS_DIR)
    listed = await catalog.list_catalog(user_id="u1", limit=2)
    assert len(listed) == 2 and all(m.description for m in listed)
    skills = await catalog.get_by_names("u1", ["drug-interaction", "ghost"])
    assert [s.name for s in skills] == ["drug-interaction"]


def test_folder_name_mismatch_fails_loudly(tmp_path: Path) -> None:
    (tmp_path / "wrong-folder").mkdir()
    (tmp_path / "wrong-folder" / "SKILL.md").write_text(
        "---\nname: drug-monograph\ndescription: d\n---\nx", encoding="utf-8"
    )
    with pytest.raises(SkillParseError, match="wrong-folder: Directory name"):
        load_system_skills(tmp_path)
