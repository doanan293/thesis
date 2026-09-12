import uuid
from collections.abc import AsyncGenerator

import pytest
from sqlalchemy import text

from pharma_agent.domain.skill.models import Skill
from pharma_agent.infrastructure.persistence.postgres.database import Database
from pharma_agent.infrastructure.persistence.postgres.skill_repository import (
    PostgresSkillRepository,
)
from pharma_agent.infrastructure.persistence.postgres.tables import UserTable

pytestmark = pytest.mark.integration


@pytest.fixture
async def database(migrated_dsn: str) -> AsyncGenerator[Database]:
    db = Database(migrated_dsn, pool_size=2)
    async with db.engine.begin() as connection:
        await connection.execute(text('TRUNCATE "user", skills CASCADE'))
    yield db
    await db.dispose()


async def make_user(database: Database, email: str) -> str:
    user_id = uuid.uuid4()
    async with database.sessions.begin() as session:
        session.add(UserTable(id=user_id, email=email, hashed_password="x"))
    return user_id.hex


def skill(skill_id: str, owner: str | None = None, version: str = "v1") -> Skill:
    return Skill(
        skill_id=skill_id,
        owner_user_id=owner,
        name=skill_id.title(),
        description=f"d {version}",
        search_guidance="s",
        answer_guidance="a",
        version=version,
    )


async def test_system_upsert_is_idempotent_and_updates_changed_versions(
    database: Database,
) -> None:
    repo = PostgresSkillRepository(database.sessions)
    await repo.upsert_system([skill("drug-monograph"), skill("plain-language")])
    await repo.upsert_system(
        [skill("drug-monograph", version="v2"), skill("plain-language")]
    )
    catalog = await repo.list_catalog(None, limit=10)
    assert sorted(m.skill_id for m in catalog) == ["drug-monograph", "plain-language"]
    loaded = await repo.get_by_ids(["drug-monograph"])
    assert (
        loaded[0].version == "v2"
        and loaded[0].description == "d v2"
        and loaded[0].is_system
    )


async def test_catalog_mixes_system_and_own_enabled_skills(database: Database) -> None:
    repo = PostgresSkillRepository(database.sessions)
    owner = await make_user(database, "a@example.com")
    other = await make_user(database, "b@example.com")
    await repo.upsert_system([skill("drug-monograph")])
    await repo.create(skill("my-notes-ab12cd", owner))
    await repo.create(skill("theirs-ab12cd", other))
    disabled = await repo.set_enabled(owner, "my-notes-ab12cd", False)
    assert disabled is not None and disabled.enabled is False
    assert [m.skill_id for m in await repo.list_catalog(owner, limit=10)] == [
        "drug-monograph"
    ]
    await repo.set_enabled(owner, "my-notes-ab12cd", True)
    assert sorted(m.skill_id for m in await repo.list_catalog(owner, limit=10)) == [
        "drug-monograph",
        "my-notes-ab12cd",
    ]
    assert [s.skill_id for s in await repo.list_for_user(owner)] == ["my-notes-ab12cd"]
    assert await repo.get_owned(other, "my-notes-ab12cd") is None
    assert await repo.set_enabled(other, "my-notes-ab12cd", False) is None
    assert await repo.delete_owned(other, "my-notes-ab12cd") is False
    assert await repo.delete_owned(owner, "my-notes-ab12cd") is True
    assert (
        await repo.get_by_ids(["my-notes-ab12cd", "drug-monograph"])
        and len(await repo.get_by_ids(["my-notes-ab12cd"])) == 0
    )


async def test_limit_applies_to_catalog(database: Database) -> None:
    repo = PostgresSkillRepository(database.sessions)
    await repo.upsert_system([skill(f"s-{i}") for i in range(5)])
    assert len(await repo.list_catalog(None, limit=3)) == 3


async def test_list_system_excludes_user_skills(database: Database) -> None:
    repo = PostgresSkillRepository(database.sessions)
    owner = await make_user(database, "a@example.com")
    await repo.upsert_system([skill("b-skill"), skill("a-skill")])
    await repo.create(skill("mine-ab12cd", owner))
    assert [s.skill_id for s in await repo.list_system()] == ["a-skill", "b-skill"]
