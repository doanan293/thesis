import pytest

from pharma_agent.domain.skill.models import Skill, SkillMetadata
from pharma_agent.domain.skill.parser import SkillParseError, parse_skill_markdown
from pharma_agent.domain.skill.resolver import resolve_selected

VALID = """---
name: Tương tác thuốc
description: Dùng khi người dùng hỏi hai thuốc có dùng chung được không.
---

# Tương tác thuốc

## Tìm kiếm
- Tìm chuyên luận của từng thuốc, mục "Tương tác".

## Trả lời
- Nêu rõ mức độ và cơ chế, trích dẫn từng thuốc.
"""


def test_parse_valid_skill_markdown() -> None:
    parsed = parse_skill_markdown(VALID)
    assert parsed.name == "Tương tác thuốc"
    assert parsed.description.startswith("Dùng khi")
    assert (
        parsed.search_guidance == '- Tìm chuyên luận của từng thuốc, mục "Tương tác".'
    )
    assert parsed.answer_guidance == "- Nêu rõ mức độ và cơ chế, trích dẫn từng thuốc."
    assert len(parsed.version) == 12


def test_parse_accepts_missing_one_section_but_not_both() -> None:
    only_search = VALID.split("## Trả lời")[0]
    assert parse_skill_markdown(only_search).answer_guidance == ""
    with pytest.raises(SkillParseError):
        parse_skill_markdown("---\nname: a\ndescription: b\n---\n\nNo sections here")


@pytest.mark.parametrize(
    "text",
    [
        "no frontmatter\n## Tìm kiếm\nx",
        "---\nname: a\n---\n## Tìm kiếm\nx",
        "---\nname: a\ndescription: b\nextra: c\n---\n## Tìm kiếm\nx",
        "---\nname: ''\ndescription: b\n---\n## Tìm kiếm\nx",
    ],
)
def test_parse_rejects_bad_frontmatter(text: str) -> None:
    with pytest.raises(SkillParseError):
        parse_skill_markdown(text)


def test_resolver_drops_unknown_ids_keeps_order_and_caps() -> None:
    catalog = [
        SkillMetadata(skill_id=f"s{i}", name=f"S{i}", description="d") for i in range(5)
    ]
    assert resolve_selected(
        catalog, ["s3", "ghost", "s1", "s3", "s0", "s4"], max_selected=3
    ) == ["s3", "s1", "s0"]


def test_skill_projections() -> None:
    skill = Skill(
        skill_id="drug-interaction",
        name="Tương tác",
        description="d",
        search_guidance="s",
        answer_guidance="a",
        version="abc",
    )
    assert skill.metadata() == SkillMetadata(
        skill_id="drug-interaction", name="Tương tác", description="d"
    )
    assert skill.to_selected().answer_guidance == "a"
    with pytest.raises(ValueError):
        Skill(skill_id="Bad_Id", name="n", description="d", version="v")
