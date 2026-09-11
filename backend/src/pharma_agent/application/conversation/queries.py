from datetime import datetime

from pydantic import BaseModel, Field

from pharma_agent.application.errors import ConversationNotFound, InvalidInput
from pharma_agent.domain.conversation.models import (
    Citation,
    Conversation,
    InvalidTitle,
    Message,
)
from pharma_agent.domain.conversation.ports import ConversationRepository
from pharma_agent.domain.shared.clock import Clock


class ConversationView(BaseModel):
    id: str
    title: str
    turn_count: int
    created_at: datetime
    updated_at: datetime

    @classmethod
    def of(cls, conversation: Conversation) -> "ConversationView":
        return cls(
            id=conversation.conversation_id,
            title=conversation.title,
            turn_count=conversation.turn_count,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
        )


class MessageView(BaseModel):
    id: str
    role: str
    content: str
    status: str
    citations: list[Citation] = Field(default_factory=list)
    phases: list[str] = Field(default_factory=list)
    created_at: datetime

    @classmethod
    def of(cls, message: Message) -> "MessageView":
        return cls(
            id=message.message_id,
            role=message.role.value,
            content=message.content,
            status=message.status,
            citations=list(message.citations),
            phases=list(message.phases),
            created_at=message.created_at,
        )


class ConversationQueries:
    def __init__(self, conversations: ConversationRepository, clock: Clock) -> None:
        self._conversations = conversations
        self._clock = clock

    async def list_conversations(
        self, user_id: str, *, limit: int, before: datetime | None = None
    ) -> list[ConversationView]:
        rows = await self._conversations.list_for_user(
            user_id, limit=limit, before=before
        )
        return [ConversationView.of(row) for row in rows]

    async def get_conversation(
        self, user_id: str, conversation_id: str
    ) -> ConversationView:
        return ConversationView.of(await self._owned(user_id, conversation_id))

    async def list_messages(
        self,
        user_id: str,
        conversation_id: str,
        *,
        limit: int,
        before: datetime | None = None,
    ) -> list[MessageView]:
        await self._owned(user_id, conversation_id)
        rows = await self._conversations.messages(
            conversation_id, limit=limit, before=before
        )
        return [MessageView.of(row) for row in rows]

    async def rename(
        self, user_id: str, conversation_id: str, title: str
    ) -> ConversationView:
        conversation = await self._owned(user_id, conversation_id)
        try:
            conversation.rename(title, now=self._clock.now())
        except InvalidTitle as exc:
            raise InvalidInput(str(exc)) from exc
        await self._conversations.update_title(conversation)
        return ConversationView.of(conversation)

    async def delete(self, user_id: str, conversation_id: str) -> None:
        if not await self._conversations.delete(user_id, conversation_id):
            raise ConversationNotFound(conversation_id)

    async def _owned(self, user_id: str, conversation_id: str) -> Conversation:
        conversation = await self._conversations.get(user_id, conversation_id)
        if conversation is None:
            raise ConversationNotFound(conversation_id)
        return conversation
