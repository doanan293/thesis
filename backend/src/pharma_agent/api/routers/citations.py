from typing import Annotated

from fastapi import APIRouter, Depends, Path

from pharma_agent.api.deps import ContainerDep, UserIdDependency
from pharma_agent.api.problems import problem_responses
from pharma_agent.api.schemas import MESSAGE_ID_PATTERN
from pharma_agent.application.conversation.citations import CitationDetail

MessageId = Annotated[str, Path(pattern=MESSAGE_ID_PATTERN)]
CitationIndex = Annotated[int, Path(ge=1, le=999)]


def build_citations_router(current_user_id: UserIdDependency) -> APIRouter:
    router = APIRouter(
        prefix="/messages",
        tags=["citations"],
        responses=problem_responses(401, 404, 422, 503),
    )
    UserId = Annotated[str, Depends(current_user_id)]

    @router.get("/{message_id}/citations/{index}", response_model=CitationDetail)
    async def get_message_citation(
        message_id: MessageId,
        index: CitationIndex,
        user_id: UserId,
        container: ContainerDep,
    ) -> CitationDetail:
        return await container.queries.get_citation(user_id, message_id, index)

    return router
