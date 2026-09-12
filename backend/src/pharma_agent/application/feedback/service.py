import logging
from datetime import datetime

from pydantic import BaseModel

from pharma_agent.application.errors import ApplicationError, InvalidInput
from pharma_agent.application.tracing import ScoreSink
from pharma_agent.domain.conversation.models import MessageRole
from pharma_agent.domain.conversation.ports import ConversationRepository
from pharma_agent.domain.feedback.models import Feedback, InvalidNote, Rating
from pharma_agent.domain.feedback.ports import FeedbackRepository
from pharma_agent.domain.shared.clock import Clock

logger = logging.getLogger(__name__)


class MessageNotFound(ApplicationError):
    code = "MESSAGE_NOT_FOUND"


class FeedbackView(BaseModel):
    message_id: str
    rating: Rating
    note: str
    created_at: datetime

    @classmethod
    def of(cls, feedback: Feedback) -> "FeedbackView":
        return cls(
            message_id=feedback.message_id,
            rating=feedback.rating,
            note=feedback.note,
            created_at=feedback.created_at,
        )


class FeedbackService:
    def __init__(
        self,
        conversations: ConversationRepository,
        feedback: FeedbackRepository,
        sink: ScoreSink,
        clock: Clock,
    ) -> None:
        self._conversations = conversations
        self._feedback = feedback
        self._sink = sink
        self._clock = clock

    async def submit(
        self, *, user_id: str, message_id: str, rating: str, note: str
    ) -> FeedbackView:
        message = await self._conversations.get_message(user_id, message_id)
        if message is None or message.role is not MessageRole.ASSISTANT:
            raise MessageNotFound(message_id)
        try:
            rating_value = Rating(rating)
        except ValueError as exc:
            raise InvalidInput("rating must be 'up' or 'down'") from exc
        try:
            feedback = Feedback.create(
                user_id=user_id,
                message_id=message_id,
                rating=rating_value,
                note=note,
                now=self._clock.now(),
            )
        except InvalidNote as exc:
            raise InvalidInput(str(exc)) from exc
        await self._feedback.save(feedback)
        if message.run_id:
            try:
                self._sink.record_feedback(
                    run_id=message.run_id, rating=rating_value.value, note=feedback.note
                )
            except Exception:
                logger.exception("failed to record feedback score for %s", message_id)
        return FeedbackView.of(feedback)
