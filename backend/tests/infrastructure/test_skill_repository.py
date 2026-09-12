import uuid
from collections.abc import AsyncGenerator

import pytest
from sqlalchemy import text

from pharma_agent.domain.skill.models import DuplicateSkillName, Skill
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


def skill(name: str, owner: str | None = None, description: str = "d") -> Skill:
    return Skill.from_markdown(
        f"---\nname: {name}\ndescription: {description}\n---\n\n# {name.title()}\n",
        owner_user_id=owner,
    )


async def test_replace_system_inserts_updates_and_removes(database: Database) -> None:
    repo = PostgresSkillRepository(database.sessions)
    await repo.replace_system([skill("drug-monograph"), skill("plain-language")])
    await repo.replace_system(
        [skill("drug-monograph", description="d v2"), skill("drug-interaction")]
    )
    assert [s.name for s in await repo.list_system()] == [
        "drug-interaction",
        "drug-monograph",
    ]
    loaded = await repo.get_by_names(None, ["drug-monograph"])
    assert loaded[0].description == "d v2" and loaded[0].is_system
    assert loaded[0].title == "Drug-Monograph"


async def test_names_are_unique_per_owner_and_system_wins(database: Database) -> None:
    repo = PostgresSkillRepository(database.sessions)
    owner = await make_user(database, "a@example.com")
    other = await make_user(database, "b@example.com")
    await repo.replace_system([skill("drug-monograph")])
    await repo.create(skill("my-notes", owner))
    await repo.create(skill("my-notes", other))
    with pytest.raises(DuplicateSkillName):
        await repo.create(skill("my-notes", owner))
    await repo.create(skill("drug-monograph", owner, description="shadow"))

    catalog = await repo.list_catalog(owner, limit=10)
    assert [(m.name, m.description) for m in catalog] == [
        ("drug-monograph", "d"),
        ("my-notes", "d"),
    ]
    fetched = await repo.get_by_names(owner, ["drug-monograph", "my-notes"])
    assert sorted((s.name, s.is_system) for s in fetched) == [
        ("drug-monograph", True),
        ("my-notes", False),
    ]


async def test_enable_disable_delete_are_owner_scoped(database: Database) -> None:
    repo = PostgresSkillRepository(database.sessions)
    owner = await make_user(database, "a@example.com")
    other = await make_user(database, "b@example.com")
    await repo.create(skill("my-notes", owner))

    disabled = await repo.set_enabled(owner, "my-notes", False)
    assert disabled is not None and disabled.enabled is False
    assert await repo.list_catalog(owner, limit=10) == []
    assert await repo.set_enabled(other, "my-notes", True) is None
    assert await repo.delete_owned(other, "my-notes") is False
    assert [s.name for s in await repo.list_for_user(owner)] == ["my-notes"]
    assert await repo.delete_owned(owner, "my-notes") is True
    assert await repo.list_for_user(owner) == []


async def test_limit_applies_to_catalog(database: Database) -> None:
    repo = PostgresSkillRepository(database.sessions)
    await repo.replace_system([skill(f"s-{i}") for i in range(5)])
    assert len(await repo.list_catalog(None, limit=3)) == 3
