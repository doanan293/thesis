from pathlib import Path

import pytest

from pharma_agent.infrastructure.skills.filesystem_catalog import (
    FileSystemSkillCatalog,
    load_system_skills,
)

SKILLS_DIR = Path(__file__).resolve().parents[2] / "skills"
EXPECTED_IDS = {
    "brand-to-generic",
    "dosing-by-population",
    "drug-interaction",
    "drug-monograph",
    "plain-language",
}


def test_repo_skills_load_and_have_both_sections() -> None:
    skills = load_system_skills(SKILLS_DIR)
    assert {s.skill_id for s in skills} == EXPECTED_IDS
    for skill in skills:
        assert skill.owner_user_id is None and skill.enabled
        assert skill.search_guidance and skill.answer_guidance, skill.skill_id
        assert (
            "không thay thế"
            not in (skill.search_guidance + skill.answer_guidance).lower()
        )


async def test_catalog_lists_metadata_with_limit_and_fetches_bodies() -> None:
    catalog = FileSystemSkillCatalog(SKILLS_DIR)
    listed = await catalog.list_catalog(user_id="u1", limit=2)
    assert len(listed) == 2 and all(m.description for m in listed)
    skills = await catalog.get_by_ids(["drug-interaction", "ghost"])
    assert [s.skill_id for s in skills] == ["drug-interaction"]


def test_bad_skill_file_fails_loudly(tmp_path: Path) -> None:
    (tmp_path / "Bad_Name").mkdir()
    (tmp_path / "Bad_Name" / "SKILL.md").write_text(
        "---\nname: x\ndescription: y\n---\n## Tìm kiếm\nz", encoding="utf-8"
    )
    with pytest.raises(
        ValueError
    ):  # pydantic rejects "Bad_Name" against SKILL_ID_PATTERN
        load_system_skills(tmp_path)
