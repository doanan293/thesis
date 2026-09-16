import uuid
from collections.abc import Collection, Sequence
from datetime import datetime

from pharma_agent.domain.conversation.models import (
    CitationBlock,
    Conversation,
    Message,
    Turn,
)
from pharma_agent.domain.conversation.turns import pair_turns
from pharma_agent.domain.feedback.models import Feedback
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
        self,
        user_id: str,
        *,
        limit: int,
        cursor: tuple[datetime, str] | None = None,
    ) -> list[Conversation]:
        # Hex ids compare like Postgres uuids (bytewise), so the order matches.
        rows = sorted(
            (
                row
                for row in self.rows.values()
                if row.user_id == user_id and row.turn_count > 0
            ),
            key=lambda row: (row.updated_at, row.conversation_id),
            reverse=True,
        )
        if cursor is not None:
            rows = [
                row for row in rows if (row.updated_at, row.conversation_id) < cursor
            ]
        return [row.model_copy(deep=True) for row in rows[:limit]]

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
        self,
        conversation_id: str,
        *,
        limit: int,
        cursor: tuple[datetime, str] | None = None,
    ) -> list[Message]:
        ordered = sorted(
            self.message_log.get(conversation_id, []),
            key=lambda message: (message.created_at, message.message_id),
        )
        if cursor is not None:
            ordered = [
                message
                for message in ordered
                if (message.created_at, message.message_id) < cursor
            ]
        return ordered[-limit:]

    async def get_message(self, user_id: str, message_id: str) -> Message | None:
        for conversation_id, messages in self.message_log.items():
            row = self.rows.get(conversation_id)
            if row is None or row.user_id != user_id:
                continue
            for message in messages:
                if message.message_id == message_id:
                    return message
        return None


class InMemoryFeedbackRepository:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], Feedback] = {}

    async def save(self, feedback: Feedback) -> None:
        self.rows[(feedback.user_id, feedback.message_id)] = feedback

    async def get(self, user_id: str, message_id: str) -> Feedback | None:
        return self.rows.get((user_id, message_id))

    async def for_messages(
        self, user_id: str, message_ids: Sequence[str]
    ) -> dict[str, Feedback]:
        wanted = set(message_ids)
        return {
            message_id: feedback
            for (owner, message_id), feedback in self.rows.items()
            if owner == user_id and message_id in wanted
        }


class InMemoryCitationReader:
    """Corpus-side citation reads for tests that run without Postgres."""

    def __init__(self, *current_release_ids: uuid.UUID) -> None:
        self.current: set[uuid.UUID] = set(current_release_ids)
        self.blocks: dict[tuple[str, str, int], CitationBlock] = {}

    async def current_release_ids(
        self, release_ids: Collection[uuid.UUID]
    ) -> set[uuid.UUID]:
        return {release_id for release_id in release_ids if release_id in self.current}

    async def citation_block(
        self, user_id: str, message_id: str, index: int
    ) -> CitationBlock | None:
        return self.blocks.get((user_id, message_id, index))
