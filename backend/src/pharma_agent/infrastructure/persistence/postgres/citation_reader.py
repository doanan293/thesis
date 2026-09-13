"""Stored citations read together with schema `corpus` (spec A §4.2, §5.2)."""

import uuid
from collections.abc import Collection

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pharma_agent.infrastructure.persistence.postgres.corpus_tables import (
    CollectionTable,
)


class PostgresCitationReader:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def current_release_ids(
        self, release_ids: Collection[uuid.UUID]
    ) -> set[uuid.UUID]:
        if not release_ids:
            return set()
        query = select(CollectionTable.current_release_id).where(
            CollectionTable.current_release_id.in_(list(release_ids))
        )
        async with self._sessions() as session:
            found = (await session.execute(query)).scalars().all()
        return {release_id for release_id in found if release_id is not None}
