import uuid
from collections.abc import Sequence

from sqlalchemy import ColumnElement, delete, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import aliased

from pharma_agent.domain.skill.models import DuplicateSkillName, Skill, SkillMetadata
from pharma_agent.infrastructure.persistence.postgres.tables import SkillTable

OWNER_NAME_CONSTRAINT = "uq_skills_owner_name"


def _uuid(value: str | None) -> uuid.UUID | None:
    if not value:
        return None
    try:
        return uuid.UUID(hex=value)
    except ValueError:
        return None


def _skill(row: SkillTable) -> Skill:
    return Skill.from_markdown(
        row.content,
        owner_user_id=row.owner_user_id.hex if row.owner_user_id else None,
        enabled=row.enabled,
    )


def _values(skill: Skill) -> dict[str, object]:
    return {
        "id": uuid.uuid4(),
        "owner_user_id": _uuid(skill.owner_user_id),
        "name": skill.name,
        "description": skill.description,
        "content": skill.content,
        "version": skill.version,
        "enabled": skill.enabled,
    }


class PostgresSkillRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def replace_system(self, skills: Sequence[Skill]) -> None:
        names = [skill.name for skill in skills]
        async with self._sessions.begin() as session:
            await session.execute(
                delete(SkillTable).where(
                    SkillTable.owner_user_id.is_(None), SkillTable.name.not_in(names)
                )
            )
            if not skills:
                return
            statement = insert(SkillTable).values([_values(s) for s in skills])
            statement = statement.on_conflict_do_update(
                constraint=OWNER_NAME_CONSTRAINT,
                set_={
                    "description": statement.excluded.description,
                    "content": statement.excluded.content,
                    "version": statement.excluded.version,
                    "updated_at": func.now(),
                },
                where=SkillTable.version != statement.excluded.version,
            )
            await session.execute(statement)

    async def create(self, skill: Skill) -> None:
        try:
            async with self._sessions.begin() as session:
                session.add(SkillTable(**_values(skill)))
        except IntegrityError as exc:
            raise DuplicateSkillName(skill.name) from exc

    async def list_catalog(
        self, user_id: str | None, limit: int
    ) -> list[SkillMetadata]:
        rows = await self._visible(user_id, names=None, limit=limit)
        return [_skill(row).metadata() for row in rows]

    async def get_by_names(
        self, user_id: str | None, names: Sequence[str]
    ) -> list[Skill]:
        if not names:
            return []
        rows = await self._visible(user_id, names=names, limit=None)
        return [_skill(row) for row in rows]

    async def list_system(self) -> list[Skill]:
        return await self._list(SkillTable.owner_user_id.is_(None))

    async def list_for_user(self, user_id: str) -> list[Skill]:
        owner = _uuid(user_id)
        if owner is None:
            return []
        return await self._list(SkillTable.owner_user_id == owner)

    async def set_enabled(self, user_id: str, name: str, enabled: bool) -> Skill | None:
        owner = _uuid(user_id)
        if owner is None:
            return None
        async with self._sessions.begin() as session:
            row = (
                await session.execute(
                    update(SkillTable)
                    .where(SkillTable.owner_user_id == owner, SkillTable.name == name)
                    .values(enabled=enabled, updated_at=func.now())
                    .returning(SkillTable)
                )
            ).scalar_one_or_none()
            return _skill(row) if row is not None else None

    async def delete_owned(self, user_id: str, name: str) -> bool:
        owner = _uuid(user_id)
        if owner is None:
            return False
        async with self._sessions.begin() as session:
            result = await session.execute(
                delete(SkillTable)
                .where(SkillTable.owner_user_id == owner, SkillTable.name == name)
                .returning(SkillTable.id)
            )
            return result.first() is not None

    async def _list(self, condition: ColumnElement[bool]) -> list[Skill]:
        query = select(SkillTable).where(condition).order_by(SkillTable.name)
        async with self._sessions() as session:
            rows = (await session.execute(query)).scalars().all()
        return [_skill(row) for row in rows]

    async def _visible(
        self, user_id: str | None, *, names: Sequence[str] | None, limit: int | None
    ) -> list[SkillTable]:
        """Enabled skills visible to the user, one per name, system skills first."""
        visible = SkillTable.owner_user_id.is_(None)
        owner = _uuid(user_id)
        if owner is not None:
            visible = or_(visible, SkillTable.owner_user_id == owner)
        one_per_name = (
            select(SkillTable)
            .where(visible, SkillTable.enabled.is_(True))
            .distinct(SkillTable.name)
            .order_by(SkillTable.name, SkillTable.owner_user_id.is_not(None))
        )
        if names is not None:
            one_per_name = one_per_name.where(SkillTable.name.in_(list(names)))
        row = aliased(SkillTable, one_per_name.subquery())
        query = select(row).order_by(row.owner_user_id.is_not(None), row.name)
        if limit is not None:
            query = query.limit(limit)
        async with self._sessions() as session:
            return list((await session.execute(query)).scalars().all())
