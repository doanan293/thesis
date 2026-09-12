from typing import Protocol

from pharma_agent.domain.feedback.models import Feedback


class FeedbackRepository(Protocol):
    async def save(self, feedback: Feedback) -> None:
        """Insert, or replace the user's earlier feedback on the same message."""
        ...

    async def get(self, user_id: str, message_id: str) -> Feedback | None: ...
