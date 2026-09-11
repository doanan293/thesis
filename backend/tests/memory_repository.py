from collections.abc import Sequence
from datetime import datetime

from pharma_agent.domain.conversation.models import Conversation, Message, Turn
from pharma_agent.domain.conversation.turns import pair_turns
from pharma_agent.domain.retrieval.audit import RetrievalRunRecord


class InMemoryConversationRepository:
    """Behaves like the Postgres repository, including owner scoping and SQL-side turn counting."""

    def __init__(self) -> None:
        self.rows: dict[str, Conversation] = {}
        self.message_log: dict[str, list[Message]] = {}
        self.audit: dict[str, list[RetrievalRunRecord]] = {}
        self.fail_append = False

    async def create(self, conversation: Conversation) -> None:
        self.rows[conversation.conversation_id] = conversation.model_copy(deep=True)
        self.message_log[conversation.conversation_id] = []

    async def get(self, user_id: str, conversation_id: str) -> Conversation | None:
        row = self.rows.get(conversation_id)
        return (
            row.model_copy(deep=True)
            if row is not None and row.user_id == user_id
            else None
        )

    async def list_for_user(
        self, user_id: str, *, limit: int, before: datetime | None = None
    ) -> list[Conversation]:
        rows = [
            row.model_copy(deep=True)
            for row in self.rows.values()
            if row.user_id == user_id and (before is None or row.updated_at < before)
        ]
        return sorted(rows, key=lambda row: row.updated_at, reverse=True)[:limit]

    async def update_title(self, conversation: Conversation) -> None:
        row = self.rows[conversation.conversation_id]
        row.title, row.updated_at = conversation.title, conversation.updated_at

    async def update_summary(self, conversation: Conversation) -> None:
        row = self.rows[conversation.conversation_id]
        row.summary = conversation.summary
        row.summarized_turns = conversation.summarized_turns
        row.updated_at = conversation.updated_at

    async def delete(self, user_id: str, conversation_id: str) -> bool:
        row = self.rows.get(conversation_id)
        if row is None or row.user_id != user_id:
            return False
        del self.rows[conversation_id]
        self.message_log.pop(conversation_id, None)
        return True

    async def append_turn(
        self,
        conversation: Conversation,
        user_message: Message,
        assistant_message: Message,
        audit: Sequence[RetrievalRunRecord],
    ) -> None:
        if self.fail_append:
            raise RuntimeError("database unavailable")
        row = self.rows.get(conversation.conversation_id)
        if row is None:
            raise LookupError(conversation.conversation_id)
        row.turn_count += 1
        row.updated_at = conversation.updated_at
        self.message_log[conversation.conversation_id].extend(
            [user_message, assistant_message]
        )
        self.audit[assistant_message.message_id] = list(audit)

    async def recent_turns(self, conversation_id: str, limit: int) -> list[Turn]:
        return pair_turns(self.message_log.get(conversation_id, []))[-limit:]

    async def turns_since(self, conversation_id: str, skip: int) -> list[Turn]:
        return pair_turns(self.message_log.get(conversation_id, [])[skip * 2 :])

    async def messages(
        self, conversation_id: str, *, limit: int, before: datetime | None = None
    ) -> list[Message]:
        items = [
            message
            for message in self.message_log.get(conversation_id, [])
            if before is None or message.created_at < before
        ]
        return items[-limit:]
