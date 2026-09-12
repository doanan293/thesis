from pathlib import Path

import pytest

from pharma_agent.application.errors import InvalidInput
from pharma_agent.application.skill.service import (
    MAX_SKILL_BYTES,
    SkillNameTaken,
    SkillNotFound,
    SkillService,
)
from tests.memory_repository import InMemorySkillRepository

OWNER, STRANGER = "a" * 32, "b" * 32
SKILLS_DIR = Path(__file__).resolve().parents[2] / "skills"
VALID_TEXT = (
    "---\n"
    "name: ghi-chu-thuoc-bo\n"
    "description: Ghi chú về thuốc bổ. Dùng khi hỏi về vitamin và khoáng chất.\n"
    "---\n\n"
    "# Ghi chú thuốc bổ\n\n"
    "Tìm mục liều.\n"
)
VALID = VALID_TEXT.encode()


def with_name(name: str) -> bytes:
    return VALID_TEXT.replace("name: ghi-chu-thuoc-bo", f"name: {name}").encode()


def service() -> tuple[SkillService, InMemorySkillRepository]:
    repo = InMemorySkillRepository()
    return SkillService(repo), repo


async def test_sync_system_loads_repo_skills() -> None:
    svc, repo = service()
    assert await svc.sync_system(SKILLS_DIR) == 5
    assert len(repo.rows) == 5 and all(s.is_system for s in repo.rows.values())


async def test_upload_keeps_the_spec_name_and_reads_the_title() -> None:
    svc, _ = service()
    created = await svc.upload(OWNER, "SKILL.md", VALID)
    assert (created.name, created.title, created.is_system, created.enabled) == (
        "ghi-chu-thuoc-bo",
        "Ghi chú thuốc bổ",
        False,
        True,
    )
    assert [s.name for s in await svc.list_for_user(OWNER)] == ["ghi-chu-thuoc-bo"]
    assert await svc.list_for_user(STRANGER) == []
    reused = await svc.upload(STRANGER, "skill.md", VALID)
    assert reused.name == "ghi-chu-thuoc-bo"


async def test_upload_name_conflicts() -> None:
    svc, _ = service()
    await svc.sync_system(SKILLS_DIR)
    await svc.upload(OWNER, "SKILL.md", VALID)
    with pytest.raises(SkillNameTaken, match="already have"):
        await svc.upload(OWNER, "SKILL.md", VALID)
    with pytest.raises(SkillNameTaken, match="system skill"):
        await svc.upload(OWNER, "SKILL.md", with_name("drug-monograph"))


async def test_upload_validation() -> None:
    svc, _ = service()
    with pytest.raises(InvalidInput, match=r"SKILL\.md"):
        await svc.upload(OWNER, "notes.txt", VALID)
    with pytest.raises(InvalidInput, match=r"64 KB"):
        await svc.upload(OWNER, "SKILL.md", b"x" * (MAX_SKILL_BYTES + 1))
    with pytest.raises(InvalidInput, match="frontmatter"):
        await svc.upload(OWNER, "SKILL.md", b"no frontmatter")
    with pytest.raises(InvalidInput, match="UTF-8"):
        await svc.upload(OWNER, "SKILL.md", b"\xff\xfe")
    with pytest.raises(InvalidInput, match="lowercase"):
        await svc.upload(OWNER, "SKILL.md", with_name("Ghi Chu"))


async def test_enable_disable_delete_are_owner_scoped() -> None:
    svc, _ = service()
    created = await svc.upload(OWNER, "SKILL.md", VALID)
    disabled = await svc.set_enabled(OWNER, created.name, False)
    assert disabled.enabled is False
    with pytest.raises(SkillNotFound):
        await svc.set_enabled(STRANGER, created.name, True)
    with pytest.raises(SkillNotFound):
        await svc.delete(STRANGER, created.name)
    await svc.delete(OWNER, created.name)
    with pytest.raises(SkillNotFound):
        await svc.delete(OWNER, created.name)


async def test_list_visible_returns_system_then_own_skills() -> None:
    svc, _ = service()
    await svc.sync_system(SKILLS_DIR)
    await svc.upload(OWNER, "SKILL.md", VALID)
    await svc.upload(STRANGER, "SKILL.md", with_name("cua-nguoi-khac"))

    visible = await svc.list_visible(OWNER)

    system_names = [view.name for view in visible if view.is_system]
    assert len(system_names) == 5 and system_names == sorted(system_names)
    assert [view.name for view in visible if not view.is_system] == ["ghi-chu-thuoc-bo"]
    assert all(view.is_system for view in visible[:5])
