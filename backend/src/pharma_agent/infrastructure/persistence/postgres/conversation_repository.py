import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pharma_agent.domain.conversation.models import (
    Citation,
    Conversation,
    Message,
    MessageRole,
    Turn,
)
from pharma_agent.domain.conversation.turns import pair_turns
from pharma_agent.domain.retrieval.audit import RetrievalRunRecord
from pharma_agent.infrastructure.persistence.postgres.tables import (
    ConversationTable,
    MessageTable,
    RetrievalHitTable,
    RetrievalRunTable,
)


class ConversationRowMissing(LookupError):
    """append_turn was called for a conversation that is not in the database."""


@dataclass(frozen=True)
class AuditContext:
    corpus_version: str
    embedding_model: str
    retriever_config: dict[str, Any] = field(default_factory=dict)


def _uuid(value: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(hex=value)
    except ValueError:
        return None


def _conversation(row: ConversationTable) -> Conversation:
    return Conversation(
        conversation_id=row.id.hex,
        user_id=row.user_id.hex,
        title=row.title,
        summary=row.summary,
        turn_count=row.turn_count,
        summarized_turns=row.summarized_turns,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _message(row: MessageTable) -> Message:
    return Message(
        message_id=row.id.hex,
        conversation_id=row.conversation_id.hex,
        role=MessageRole(row.role),
        content=row.content,
        status=row.status,
        citations=[Citation.model_validate(item) for item in row.citations],
        phases=list(row.phases),
        usage=dict(row.usage),
        run_id=row.run_id,
        created_at=row.created_at,
    )


def _message_row(message: Message) -> MessageTable:
    return MessageTable(
        id=uuid.UUID(hex=message.message_id),
        conversation_id=uuid.UUID(hex=message.conversation_id),
        role=message.role.value,
        content=message.content,
        status=message.status,
        citations=[citation.model_dump(mode="json") for citation in message.citations],
        phases=list(message.phases),
        usage=dict(message.usage),
        run_id=message.run_id,
        created_at=message.created_at,
    )


class PostgresConversationRepository:
    def __init__(
        self, sessions: async_sessionmaker[AsyncSession], audit_context: AuditContext
    ) -> None:
        self._sessions = sessions
        self._audit_context = audit_context

    async def create(self, conversation: Conversation) -> None:
        async with self._sessions.begin() as session:
            session.add(
                ConversationTable(
                    id=uuid.UUID(hex=conversation.conversation_id),
                    user_id=uuid.UUID(hex=conversation.user_id),
                    title=conversation.title,
                    summary=conversation.summary,
                    turn_count=conversation.turn_count,
                    summarized_turns=conversation.summarized_turns,
                    created_at=conversation.created_at,
                    updated_at=conversation.updated_at,
                )
            )

    async def get(self, user_id: str, conversation_id: str) -> Conversation | None:
        owner, key = _uuid(user_id), _uuid(conversation_id)
        if owner is None or key is None:
            return None
        async with self._sessions() as session:
            row = (
                await session.execute(
                    select(ConversationTable).where(
                        ConversationTable.id == key, ConversationTable.user_id == owner
                    )
                )
            ).scalar_one_or_none()
        return _conversation(row) if row is not None else None

    async def list_for_user(
        self, user_id: str, *, limit: int, before: datetime | None = None
    ) -> list[Conversation]:
        owner = _uuid(user_id)
        if owner is None:
            return []
        query = select(ConversationTable).where(ConversationTable.user_id == owner)
        if before is not None:
            query = query.where(ConversationTable.updated_at < before)
        query = query.order_by(ConversationTable.updated_at.desc()).limit(limit)
        async with self._sessions() as session:
            rows = (await session.execute(query)).scalars().all()
        return [_conversation(row) for row in rows]

    async def update_title(self, conversation: Conversation) -> None:
        await self._update(
            conversation, title=conversation.title, updated_at=conversation.updated_at
        )

    async def update_summary(self, conversation: Conversation) -> None:
        await self._update(
            conversation,
            summary=conversation.summary,
            summarized_turns=conversation.summarized_turns,
            updated_at=conversation.updated_at,
        )

    async def delete(self, user_id: str, conversation_id: str) -> bool:
        owner, key = _uuid(user_id), _uuid(conversation_id)
        if owner is None or key is None:
            return False
        async with self._sessions.begin() as session:
            result = await session.execute(
                delete(ConversationTable)
                .where(ConversationTable.id == key, ConversationTable.user_id == owner)
                .returning(ConversationTable.id)
            )
            return result.first() is not None

    async def append_turn(
        self,
        conversation: Conversation,
        user_message: Message,
        assistant_message: Message,
        audit: Sequence[RetrievalRunRecord],
    ) -> None:
        key = uuid.UUID(hex=conversation.conversation_id)
        assistant_id = uuid.UUID(hex=assistant_message.message_id)
        async with self._sessions.begin() as session:
            updated = await session.execute(
                update(ConversationTable)
                .where(ConversationTable.id == key)
                .values(
                    turn_count=ConversationTable.turn_count + 1,
                    updated_at=conversation.updated_at,
                )
                .returning(ConversationTable.id)
            )
            if updated.first() is None:
                raise ConversationRowMissing(conversation.conversation_id)
            session.add_all(
                [_message_row(user_message), _message_row(assistant_message)]
            )
            await session.flush()
            for record in audit:
                run_row = RetrievalRunTable(
                    message_id=assistant_id,
                    conversation_id=key,
                    run_id=assistant_message.run_id or "",
                    round=record.round,
                    query_text=record.query_text,
                    corpus_version=self._audit_context.corpus_version,
                    embedding_model=self._audit_context.embedding_model,
                    retriever_config=dict(self._audit_context.retriever_config),
                    created_at=assistant_message.created_at,
                )
                session.add(run_row)
                await session.flush()
                session.add_all(
                    RetrievalHitTable(retrieval_run_id=run_row.id, **hit.model_dump())
                    for hit in record.hits
                )

    async def recent_turns(self, conversation_id: str, limit: int) -> list[Turn]:
        messages = await self.messages(conversation_id, limit=limit * 2)
        return pair_turns(messages)[-limit:]

    async def turns_since(self, conversation_id: str, skip: int) -> list[Turn]:
        key = _uuid(conversation_id)
        if key is None:
            return []
        query = (
            select(MessageTable)
            .where(MessageTable.conversation_id == key)
            .order_by(MessageTable.created_at)
            .offset(skip * 2)
        )
        async with self._sessions() as session:
            rows = (await session.execute(query)).scalars().all()
        return pair_turns([_message(row) for row in rows])

    async def messages(
        self, conversation_id: str, *, limit: int, before: datetime | None = None
    ) -> list[Message]:
        key = _uuid(conversation_id)
        if key is None:
            return []
        query = select(MessageTable).where(MessageTable.conversation_id == key)
        if before is not None:
            query = query.where(MessageTable.created_at < before)
        query = query.order_by(MessageTable.created_at.desc()).limit(limit)
        async with self._sessions() as session:
            rows = (await session.execute(query)).scalars().all()
        return [_message(row) for row in reversed(rows)]

    async def get_message(self, user_id: str, message_id: str) -> Message | None:
        owner, key = _uuid(user_id), _uuid(message_id)
        if owner is None or key is None:
            return None
        query = (
            select(MessageTable)
            .join(
                ConversationTable, ConversationTable.id == MessageTable.conversation_id
            )
            .where(MessageTable.id == key, ConversationTable.user_id == owner)
        )
        async with self._sessions() as session:
            row = (await session.execute(query)).scalar_one_or_none()
        return _message(row) if row is not None else None

    async def _update(self, conversation: Conversation, **values: object) -> None:
        async with self._sessions.begin() as session:
            await session.execute(
                update(ConversationTable)
                .where(
                    ConversationTable.id == uuid.UUID(hex=conversation.conversation_id)
                )
                .values(**values)
            )
