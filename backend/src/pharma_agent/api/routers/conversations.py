from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query, Response

from pharma_agent.api.deps import ContainerDep, UserIdDependency
from pharma_agent.api.schemas import CONVERSATION_ID_PATTERN, RenameConversationRequest
from pharma_agent.application.conversation.queries import (
    ConversationPage,
    ConversationView,
    MessagePage,
)

MAX_PAGE_SIZE = 100
MAX_CURSOR_LENGTH = 512

ConversationId = Annotated[str, Path(pattern=CONVERSATION_ID_PATTERN)]
Cursor = Annotated[
    str | None,
    Query(
        max_length=MAX_CURSOR_LENGTH,
        description="`next_cursor` of the previous page; omit for the first page",
    ),
]


def build_conversations_router(current_user_id: UserIdDependency) -> APIRouter:
    router = APIRouter(prefix="/conversations", tags=["conversations"])
    UserId = Annotated[str, Depends(current_user_id)]

    @router.get("", response_model=ConversationPage)
    async def list_conversations(
        user_id: UserId,
        container: ContainerDep,
        limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 20,
        cursor: Cursor = None,
    ) -> ConversationPage:
        return await container.queries.list_conversations(
            user_id, limit=limit, cursor=cursor
        )

    @router.get("/{conversation_id}", response_model=ConversationView)
    async def get_conversation(
        conversation_id: ConversationId, user_id: UserId, container: ContainerDep
    ) -> ConversationView:
        return await container.queries.get_conversation(user_id, conversation_id)

    @router.get("/{conversation_id}/messages", response_model=MessagePage)
    async def list_messages(
        conversation_id: ConversationId,
        user_id: UserId,
        container: ContainerDep,
        limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 30,
        cursor: Cursor = None,
    ) -> MessagePage:
        return await container.queries.list_messages(
            user_id, conversation_id, limit=limit, cursor=cursor
        )

    @router.patch("/{conversation_id}", response_model=ConversationView)
    async def rename_conversation(
        conversation_id: ConversationId,
        body: RenameConversationRequest,
        user_id: UserId,
        container: ContainerDep,
    ) -> ConversationView:
        return await container.queries.rename(user_id, conversation_id, body.title)

    @router.delete("/{conversation_id}", status_code=204)
    async def delete_conversation(
        conversation_id: ConversationId, user_id: UserId, container: ContainerDep
    ) -> Response:
        await container.queries.delete(user_id, conversation_id)
        return Response(status_code=204)

    return router
