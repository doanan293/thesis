from pathlib import Path

import pytest

from pharma_agent.application.errors import InvalidInput
from pharma_agent.application.skill.service import (
    MAX_SKILL_BYTES,
    SkillNotFound,
    SkillService,
)
from tests.memory_repository import InMemorySkillRepository

OWNER, STRANGER = "a" * 32, "b" * 32
SKILLS_DIR = Path(__file__).resolve().parents[2] / "skills"
VALID = b"""---
name: Ghi ch\xc3\xba c\xe1\xbb\xa7a t\xc3\xb4i
description: D\xc3\xb9ng khi h\xe1\xbb\x8fi v\xe1\xbb\x81 thu\xe1\xbb\x91c b\xe1\xbb\x95 sung.
---

## T\xc3\xacm ki\xe1\xba\xbfm
- T\xc3\xacm m\xe1\xbb\xa5c li\xe1\xbb\x81u.
"""


def service(
    repo: InMemorySkillRepository | None = None,
) -> tuple[SkillService, InMemorySkillRepository]:
    repo = repo or InMemorySkillRepository()
    return SkillService(repo), repo


async def test_sync_system_loads_repo_skills() -> None:
    svc, repo = service()
    await svc.sync_system(SKILLS_DIR)
    assert len(repo.rows) == 5 and all(s.is_system for s in repo.rows.values())


async def test_upload_creates_slugged_unique_skill_and_lists_it() -> None:
    svc, _ = service()
    first = await svc.upload(OWNER, "SKILL.md", VALID)
    second = await svc.upload(OWNER, "skill.md", VALID)
    assert (
        first.id.startswith("ghi-chu-cua-toi-")
        and len(first.id) == len("ghi-chu-cua-toi-") + 6
    )
    assert first.id != second.id and first.is_system is False and first.enabled is True
    listed = await svc.list_for_user(OWNER)
    assert {s.id for s in listed} == {first.id, second.id}
    assert await svc.list_for_user(STRANGER) == []


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


async def test_enable_disable_delete_are_owner_scoped() -> None:
    svc, _ = service()
    created = await svc.upload(OWNER, "SKILL.md", VALID)
    disabled = await svc.set_enabled(OWNER, created.id, False)
    assert disabled.enabled is False
    with pytest.raises(SkillNotFound):
        await svc.set_enabled(STRANGER, created.id, True)
    with pytest.raises(SkillNotFound):
        await svc.delete(STRANGER, created.id)
    await svc.delete(OWNER, created.id)
    with pytest.raises(SkillNotFound):
        await svc.delete(OWNER, created.id)
