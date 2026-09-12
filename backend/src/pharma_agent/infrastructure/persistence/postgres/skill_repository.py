import uuid
from collections.abc import Sequence

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pharma_agent.domain.skill.models import Skill, SkillMetadata
from pharma_agent.infrastructure.persistence.postgres.tables import SkillTable


def _uuid(value: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(hex=value)
    except ValueError:
        return None


def _skill(row: SkillTable) -> Skill:
    return Skill(
        skill_id=row.skill_id,
        owner_user_id=row.owner_user_id.hex if row.owner_user_id else None,
        name=row.name,
        description=row.description,
        search_guidance=row.search_guidance,
        answer_guidance=row.answer_guidance,
        version=row.version,
        enabled=row.enabled,
    )


def _values(skill: Skill) -> dict[str, object]:
    return {
        "skill_id": skill.skill_id,
        "owner_user_id": uuid.UUID(hex=skill.owner_user_id)
        if skill.owner_user_id
        else None,
        "name": skill.name,
        "description": skill.description,
        "search_guidance": skill.search_guidance,
        "answer_guidance": skill.answer_guidance,
        "version": skill.version,
        "enabled": skill.enabled,
    }


class PostgresSkillRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def upsert_system(self, skills: Sequence[Skill]) -> None:
        if not skills:
            return
        statement = insert(SkillTable).values([_values(s) for s in skills])
        statement = statement.on_conflict_do_update(
            index_elements=[SkillTable.skill_id],
            set_={
                "name": statement.excluded.name,
                "description": statement.excluded.description,
                "search_guidance": statement.excluded.search_guidance,
                "answer_guidance": statement.excluded.answer_guidance,
                "version": statement.excluded.version,
                "updated_at": func.now(),
            },
            where=SkillTable.version != statement.excluded.version,
        )
        async with self._sessions.begin() as session:
            await session.execute(statement)

    async def create(self, skill: Skill) -> None:
        async with self._sessions.begin() as session:
            session.add(SkillTable(**_values(skill)))

    async def list_catalog(
        self, user_id: str | None, limit: int
    ) -> list[SkillMetadata]:
        owner = _uuid(user_id) if user_id else None
        visible = SkillTable.owner_user_id.is_(None)
        if owner is not None:
            visible = or_(visible, SkillTable.owner_user_id == owner)
        query = (
            select(SkillTable)
            .where(visible, SkillTable.enabled.is_(True))
            .order_by(SkillTable.owner_user_id.is_not(None), SkillTable.skill_id)
            .limit(limit)
        )
        async with self._sessions() as session:
            rows = (await session.execute(query)).scalars().all()
        return [_skill(row).metadata() for row in rows]

    async def get_by_ids(self, skill_ids: Sequence[str]) -> list[Skill]:
        if not skill_ids:
            return []
        async with self._sessions() as session:
            rows = (
                (
                    await session.execute(
                        select(SkillTable).where(
                            SkillTable.skill_id.in_(list(skill_ids))
                        )
                    )
                )
                .scalars()
                .all()
            )
        return [_skill(row) for row in rows]

    async def list_system(self) -> list[Skill]:
        query = (
            select(SkillTable)
            .where(SkillTable.owner_user_id.is_(None))
            .order_by(SkillTable.skill_id)
        )
        async with self._sessions() as session:
            rows = (await session.execute(query)).scalars().all()
        return [_skill(row) for row in rows]

    async def list_for_user(self, user_id: str) -> list[Skill]:
        owner = _uuid(user_id)
        if owner is None:
            return []
        async with self._sessions() as session:
            rows = (
                (
                    await session.execute(
                        select(SkillTable)
                        .where(SkillTable.owner_user_id == owner)
                        .order_by(SkillTable.name)
                    )
                )
                .scalars()
                .all()
            )
        return [_skill(row) for row in rows]

    async def get_owned(self, user_id: str, skill_id: str) -> Skill | None:
        owner = _uuid(user_id)
        if owner is None:
            return None
        async with self._sessions() as session:
            row = (
                await session.execute(
                    select(SkillTable).where(
                        SkillTable.skill_id == skill_id,
                        SkillTable.owner_user_id == owner,
                    )
                )
            ).scalar_one_or_none()
        return _skill(row) if row is not None else None

    async def set_enabled(
        self, user_id: str, skill_id: str, enabled: bool
    ) -> Skill | None:
        owner = _uuid(user_id)
        if owner is None:
            return None
        async with self._sessions.begin() as session:
            row = (
                await session.execute(
                    update(SkillTable)
                    .where(
                        SkillTable.skill_id == skill_id,
                        SkillTable.owner_user_id == owner,
                    )
                    .values(enabled=enabled, updated_at=func.now())
                    .returning(SkillTable)
                )
            ).scalar_one_or_none()
            return _skill(row) if row is not None else None

    async def delete_owned(self, user_id: str, skill_id: str) -> bool:
        owner = _uuid(user_id)
        if owner is None:
            return False
        async with self._sessions.begin() as session:
            result = await session.execute(
                delete(SkillTable)
                .where(
                    SkillTable.skill_id == skill_id, SkillTable.owner_user_id == owner
                )
                .returning(SkillTable.skill_id)
            )
            return result.first() is not None
