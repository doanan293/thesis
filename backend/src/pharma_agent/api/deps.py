from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pharma_agent.application.chat.service import ChatService
from pharma_agent.application.errors import AgentUnavailable
from pharma_agent.application.feedback.service import FeedbackService
from pharma_agent.application.skill.service import SkillService
from pharma_agent.infrastructure.auth.users import Auth
from pharma_agent.infrastructure.container import Container
from pharma_agent.infrastructure.persistence.postgres.tables import UserTable

UserIdDependency = Callable[..., Awaitable[str]]


def get_container(request: Request) -> Container:
    container = getattr(request.app.state, "container", None)
    if not isinstance(container, Container):
        raise HTTPException(status_code=503, detail="service is starting")
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


def session_factory(request: Request) -> async_sessionmaker[AsyncSession]:
    sessions = get_container(request).sessions
    if sessions is None:
        raise HTTPException(status_code=503, detail="database is not configured")
    return sessions


def require_chat(container: Container) -> ChatService:
    if container.chat is None:
        raise AgentUnavailable(
            "the agent is not configured (set PHARMA_LLM__DEFAULT__API_KEY)"
        )
    return container.chat


def require_skills(container: Container) -> SkillService:
    if container.skills is None:
        raise HTTPException(status_code=503, detail="skills are not configured")
    return container.skills


def require_feedback(container: Container) -> FeedbackService:
    if container.feedback is None:
        raise HTTPException(status_code=503, detail="feedback is not configured")
    return container.feedback


def user_id_dependency(auth: Auth) -> UserIdDependency:
    async def current_user_id(
        user: Annotated[UserTable, Depends(auth.current_active_user)],
    ) -> str:
        return user.id.hex

    return current_user_id
