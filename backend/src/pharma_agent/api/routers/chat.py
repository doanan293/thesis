import logging
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends
from starlette.background import BackgroundTask

from pharma_agent.api.deps import ContainerDep, UserIdDependency, require_chat
from pharma_agent.api.problems import problem_responses
from pharma_agent.api.schemas import ChatRequest
from pharma_agent.api.ui_stream import (
    UI_MESSAGE_STREAM_HEADERS,
    UIMessageStreamResponse,
    ui_message_stream,
)
from pharma_agent.application.chat.service import ChatTurnResult
from pharma_agent.application.conversation.ui_message import PharmaDataParts
from pharma_agent.application.memory.summarize import SummarizeConversation

logger = logging.getLogger(__name__)


async def summarize_quietly(
    summarizer: SummarizeConversation | None, user_id: str, conversation_id: str
) -> None:
    if summarizer is None:
        return
    try:
        await summarizer.run_if_needed(user_id=user_id, conversation_id=conversation_id)
    except Exception:
        logger.exception("background summary failed for %s", conversation_id)


def build_chat_router(current_user_id: UserIdDependency) -> APIRouter:
    router = APIRouter(
        prefix="/chat", tags=["chat"], responses=problem_responses(401, 404, 422, 503)
    )
    UserId = Annotated[str, Depends(current_user_id)]

    @router.post("", response_model=ChatTurnResult)
    async def chat(
        body: ChatRequest,
        user_id: UserId,
        container: ContainerDep,
        background: BackgroundTasks,
    ) -> ChatTurnResult:
        service = require_chat(container)
        result = await service.ask(
            user_id=user_id, message=body.message, conversation_id=body.conversation_id
        )
        background.add_task(
            summarize_quietly, container.summarizer, user_id, result.conversation_id
        )
        return result

    @router.post(
        "/stream",
        response_class=UIMessageStreamResponse,
        responses={
            200: {
                "model": PharmaDataParts,
                "description": (
                    "AI SDK UI Message Stream v1 (text/event-stream). The schema lists the "
                    "payload of each data-<name> part; finish metadata is MessageMetadata."
                ),
            }
        },
    )
    async def chat_stream(
        body: ChatRequest, user_id: UserId, container: ContainerDep
    ) -> UIMessageStreamResponse:
        service = require_chat(container)
        session = await service.open_turn(
            user_id=user_id, message=body.message, conversation_id=body.conversation_id
        )
        return UIMessageStreamResponse(
            ui_message_stream(session.events()),
            ping=15,
            headers=UI_MESSAGE_STREAM_HEADERS,
            background=BackgroundTask(
                summarize_quietly,
                container.summarizer,
                user_id,
                session.conversation_id,
            ),
        )

    return router
