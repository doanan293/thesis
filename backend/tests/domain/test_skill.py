from pathlib import Path

import pytest
from skills_ref.validator import validate

from pharma_agent.domain.skill.models import Skill, SkillMetadata
from pharma_agent.domain.skill.parser import SkillParseError, parse_skill_markdown
from pharma_agent.domain.skill.resolver import resolve_selected

MINIMAL = (
    "---\n"
    "name: drug-interaction\n"
    "description: Tra cứu tương tác thuốc. Dùng khi hỏi hai thuốc có dùng chung được không.\n"
    "---\n\n"
    "# Tương tác thuốc\n\n"
    "Tìm mục Tương tác của từng thuốc.\n"
)


def test_minimal_skill_parses_with_free_form_body() -> None:
    parsed = parse_skill_markdown(MINIMAL, directory_name="drug-interaction")
    assert parsed.name == "drug-interaction"
    assert parsed.instructions.startswith("# Tương tác thuốc")
    assert len(parsed.version) == 12 and parsed.metadata == {}


def test_optional_spec_fields_are_accepted() -> None:
    text = (
        "---\n"
        "name: pdf-processing\n"
        "description: Extract PDF text. Use when handling PDFs.\n"
        "license: Apache-2.0\n"
        "compatibility: Requires network access\n"
        "allowed-tools: Read\n"
        "metadata:\n"
        "  author: example-org\n"
        '  version: "1.0"\n'
        "---\n\nBody\n"
    )
    parsed = parse_skill_markdown(text)
    assert parsed.license == "Apache-2.0" and parsed.allowed_tools == "Read"
    assert parsed.compatibility == "Requires network access"
    assert parsed.metadata == {"author": "example-org", "version": "1.0"}


def test_non_ascii_names_are_rejected_for_portability() -> None:
    with pytest.raises(SkillParseError, match="a-z"):
        parse_skill_markdown("---\nname: tương-tác\ndescription: d\n---\nx")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("no frontmatter", "frontmatter"),
        ("---\nname: a\ndescription: b\n", "closed"),
        ("---\nname: Tra cứu chuyên luận thuốc\ndescription: d\n---\nx", "lowercase"),
        ("---\nname: -pdf\ndescription: d\n---\nx", "hyphen"),
        ("---\nname: pdf--processing\ndescription: d\n---\nx", "consecutive"),
        ("---\nname: " + "a" * 65 + "\ndescription: d\n---\nx", "64"),
        ("---\nname: a\n---\nx", "description"),
        ("---\nname: a\ndescription: " + "d" * 1025 + "\n---\nx", "1024"),
        ("---\nname: a\ndescription: d\nversion: 1\n---\nx", "Unexpected fields"),
        (
            "---\nname: a\ndescription: d\ncompatibility: " + "c" * 501 + "\n---\nx",
            "500",
        ),
    ],
)
def test_invalid_frontmatter_is_rejected(text: str, message: str) -> None:
    with pytest.raises(SkillParseError, match=message):
        parse_skill_markdown(text)


def test_name_must_match_the_directory() -> None:
    with pytest.raises(SkillParseError, match="must match skill name"):
        parse_skill_markdown(MINIMAL, directory_name="drug-monograph")


@pytest.mark.parametrize(
    "name",
    [
        "drug-monograph",
        "pdf2text",
        "tương-tác",
        "Tra cứu",
        "-x",
        "a--b",
        "a_b",
        "a" * 65,
    ],
)
def test_parser_never_accepts_what_the_reference_validator_rejects(
    tmp_path: Path, name: str
) -> None:
    folder = tmp_path / name
    folder.mkdir()
    text = f"---\nname: {name}\ndescription: d\n---\nx"
    (folder / "SKILL.md").write_text(text, encoding="utf-8")
    reference_accepts = validate(folder) == []
    try:
        parse_skill_markdown(text, directory_name=name)
        ours_accepts = True
    except SkillParseError:
        ours_accepts = False
    assert ours_accepts == (reference_accepts and name.isascii())


def test_skill_title_metadata_and_selection() -> None:
    skill = Skill.from_markdown(MINIMAL, directory_name="drug-interaction")
    assert skill.title == "Tương tác thuốc" and skill.is_system
    assert skill.metadata() == SkillMetadata(
        name="drug-interaction", description=skill.description
    )
    selected = skill.to_selected()
    assert selected.name == "drug-interaction" and selected.title == "Tương tác thuốc"
    assert "Tìm mục Tương tác" in selected.instructions
    untitled = Skill.from_markdown("---\nname: plain\ndescription: d\n---\nNo heading")
    assert untitled.title == "plain"


def test_resolver_drops_unknown_names_keeps_order_and_caps() -> None:
    catalog = [SkillMetadata(name=f"s{i}", description="d") for i in range(5)]
    assert resolve_selected(
        catalog, ["s3", "ghost", "s1", "s3", "s0", "s4"], max_selected=3
    ) == ["s3", "s1", "s0"]
