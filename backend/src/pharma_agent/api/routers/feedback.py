from typing import Annotated

from fastapi import APIRouter, Depends, Path

from pharma_agent.api.deps import ContainerDep, UserIdDependency, require_feedback
from pharma_agent.api.problems import problem_responses
from pharma_agent.api.schemas import MESSAGE_ID_PATTERN, FeedbackRequest
from pharma_agent.application.feedback.service import FeedbackView

MessageId = Annotated[str, Path(pattern=MESSAGE_ID_PATTERN)]


def build_feedback_router(current_user_id: UserIdDependency) -> APIRouter:
    router = APIRouter(
        prefix="/messages",
        tags=["feedback"],
        responses=problem_responses(401, 404, 422, 503),
    )
    UserId = Annotated[str, Depends(current_user_id)]

    @router.post("/{message_id}/feedback", response_model=FeedbackView, status_code=201)
    async def submit_feedback(
        message_id: MessageId,
        body: FeedbackRequest,
        user_id: UserId,
        container: ContainerDep,
    ) -> FeedbackView:
        return await require_feedback(container).submit(
            user_id=user_id, message_id=message_id, rating=body.rating, note=body.note
        )

    return router
