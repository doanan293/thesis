import uuid
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pharma_agent.domain.feedback.models import Feedback, Rating
from pharma_agent.infrastructure.persistence.postgres.tables import FeedbackTable


def _uuid(value: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(hex=value)
    except ValueError:
        return None


def _feedback(row: FeedbackTable) -> Feedback:
    return Feedback(
        feedback_id=row.feedback_id.hex,
        user_id=row.user_id.hex,
        message_id=row.message_id.hex,
        rating=Rating(row.rating),
        note=row.note,
        created_at=row.created_at,
    )


class PostgresFeedbackRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def save(self, feedback: Feedback) -> None:
        statement = insert(FeedbackTable).values(
            feedback_id=uuid.UUID(hex=feedback.feedback_id),
            user_id=uuid.UUID(hex=feedback.user_id),
            message_id=uuid.UUID(hex=feedback.message_id),
            rating=feedback.rating.value,
            note=feedback.note,
            created_at=feedback.created_at,
        )
        statement = statement.on_conflict_do_update(
            constraint="uq_feedback_user_message",
            set_={
                "rating": statement.excluded.rating,
                "note": statement.excluded.note,
                "created_at": statement.excluded.created_at,
            },
        )
        async with self._sessions.begin() as session:
            await session.execute(statement)

    async def get(self, user_id: str, message_id: str) -> Feedback | None:
        owner, key = _uuid(user_id), _uuid(message_id)
        if owner is None or key is None:
            return None
        async with self._sessions() as session:
            row = (
                await session.execute(
                    select(FeedbackTable).where(
                        FeedbackTable.user_id == owner, FeedbackTable.message_id == key
                    )
                )
            ).scalar_one_or_none()
        return _feedback(row) if row is not None else None

    async def for_messages(
        self, user_id: str, message_ids: Sequence[str]
    ) -> dict[str, Feedback]:
        owner = _uuid(user_id)
        keys = [key for value in message_ids if (key := _uuid(value)) is not None]
        if owner is None or not keys:
            return {}
        query = select(FeedbackTable).where(
            FeedbackTable.user_id == owner, FeedbackTable.message_id.in_(keys)
        )
        async with self._sessions() as session:
            rows = (await session.execute(query)).scalars().all()
        return {row.message_id.hex: _feedback(row) for row in rows}
