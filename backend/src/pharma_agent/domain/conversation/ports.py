from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from pharma_agent.domain.conversation.models import Conversation, Message, Turn
from pharma_agent.domain.retrieval.audit import RetrievalRunRecord


class ConversationRepository(Protocol):
    async def create(self, conversation: Conversation) -> None: ...

    async def get(self, user_id: str, conversation_id: str) -> Conversation | None:
        """Return the conversation only when it belongs to user_id."""
        ...

    async def list_for_user(
        self, user_id: str, *, limit: int, before: datetime | None = None
    ) -> list[Conversation]:
        """Most recently updated first."""
        ...

    async def update_title(self, conversation: Conversation) -> None:
        """Persist only title and updated_at."""
        ...

    async def update_summary(self, conversation: Conversation) -> None:
        """Persist only summary, summarized_turns and updated_at."""
        ...

    async def delete(self, user_id: str, conversation_id: str) -> bool: ...

    async def append_turn(
        self,
        conversation: Conversation,
        user_message: Message,
        assistant_message: Message,
        audit: Sequence[RetrievalRunRecord],
    ) -> None:
        """Write both messages and the audit, and increment turn_count, in one transaction."""
        ...

    async def recent_turns(self, conversation_id: str, limit: int) -> list[Turn]:
        """The last `limit` turns, oldest first."""
        ...

    async def turns_since(self, conversation_id: str, skip: int) -> list[Turn]:
        """All turns after the first `skip` turns, oldest first."""
        ...

    async def messages(
        self, conversation_id: str, *, limit: int, before: datetime | None = None
    ) -> list[Message]:
        """Newest `limit` messages strictly before `before`, returned oldest first."""
        ...

    async def get_message(self, user_id: str, message_id: str) -> Message | None:
        """A message, only if it belongs to one of the user's conversations."""
        ...
