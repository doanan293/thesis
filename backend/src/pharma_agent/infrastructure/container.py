"""Owns every long-lived resource of the HTTP service and wires application services."""

import asyncio
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import (
    AbstractAsyncContextManager,
    AsyncExitStack,
    asynccontextmanager,
    suppress,
)
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pharma_agent.application.chat.service import ChatService, MemoryPolicy
from pharma_agent.application.conversation.queries import ConversationQueries
from pharma_agent.application.feedback.service import FeedbackService
from pharma_agent.application.memory.summarize import SummarizeConversation
from pharma_agent.application.skill.service import SkillService
from pharma_agent.application.tracing import NullTracing, Tracing
from pharma_agent.domain.retrieval.ports import RetrievalError
from pharma_agent.domain.shared.clock import SystemClock
from pharma_agent.infrastructure.composition import Application, build_application
from pharma_agent.infrastructure.langgraph.checkpointer import (
    open_postgres_checkpointer,
)
from pharma_agent.infrastructure.langgraph.cleanup import delete_expired_checkpoints
from pharma_agent.infrastructure.observability.langfuse_tracing import LangfuseTracing
from pharma_agent.infrastructure.persistence.postgres.conversation_repository import (
    AuditContext,
    PostgresConversationRepository,
)
from pharma_agent.infrastructure.persistence.postgres.database import Database
from pharma_agent.infrastructure.persistence.postgres.feedback_repository import (
    PostgresFeedbackRepository,
)
from pharma_agent.infrastructure.persistence.postgres.skill_repository import (
    PostgresSkillRepository,
)
from pharma_agent.infrastructure.settings import Settings

logger = logging.getLogger(__name__)

HealthCheck = Callable[[], Awaitable[bool]]


@dataclass
class Container:
    settings: Settings
    sessions: async_sessionmaker[AsyncSession] | None
    queries: ConversationQueries
    chat: ChatService | None = None
    summarizer: SummarizeConversation | None = None
    skills: SkillService | None = None
    feedback: FeedbackService | None = None
    tracing: Tracing = field(default_factory=NullTracing)
    health_checks: dict[str, HealthCheck] = field(default_factory=dict)


ContainerFactory = Callable[[Settings], AbstractAsyncContextManager[Container]]


def _qdrant_check(agent: Application, dimension: int) -> HealthCheck:
    async def check() -> bool:
        try:
            await agent.retriever.verify_collection(dimension)
        except RetrievalError:
            return False
        return True

    return check


async def _cleanup_checkpoints(conninfo: str, retention_days: int) -> None:
    try:
        deleted = await delete_expired_checkpoints(
            conninfo, retention_days=retention_days
        )
    except Exception:
        logger.exception("checkpoint cleanup failed")
        return
    logger.info("deleted %d checkpoints older than %d days", deleted, retention_days)


async def _cancel(task: asyncio.Task[None]) -> None:
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


@asynccontextmanager
async def open_container(settings: Settings) -> AsyncGenerator[Container]:
    database = Database(
        settings.postgres.dsn,
        pool_size=settings.postgres.pool_size,
        echo=settings.postgres.echo,
    )
    clock = SystemClock()
    repository = PostgresConversationRepository(
        database.sessions,
        AuditContext(
            corpus_version=settings.retrieval.collection_alias,
            embedding_model=settings.retrieval.embedding.model,
            retriever_config=settings.retrieval.model_dump(
                mode="json", exclude={"embedding": {"api_key"}}
            ),
        ),
    )
    tracing: Tracing = (
        LangfuseTracing.from_settings(settings.langfuse)
        if settings.langfuse.enabled
        else NullTracing()
    )
    skill_repository = PostgresSkillRepository(database.sessions)
    container = Container(
        settings=settings,
        sessions=database.sessions,
        queries=ConversationQueries(repository, clock),
        skills=SkillService(skill_repository),
        feedback=FeedbackService(
            repository, PostgresFeedbackRepository(database.sessions), tracing, clock
        ),
        tracing=tracing,
        health_checks={"postgres": database.ping},
    )
    try:
        async with AsyncExitStack() as stack:
            if isinstance(tracing, LangfuseTracing):
                stack.callback(tracing.shutdown)
            if container.skills is not None:
                await container.skills.sync_system(settings.skills_dir)
            if settings.llm.configured:
                checkpointer = await stack.enter_async_context(
                    open_postgres_checkpointer(settings.postgres.conninfo)
                )
                cleanup = asyncio.create_task(
                    _cleanup_checkpoints(
                        settings.postgres.conninfo, settings.checkpoints.retention_days
                    )
                )
                stack.push_async_callback(_cancel, cleanup)
                agent = build_application(
                    settings,
                    checkpointer=checkpointer,
                    skills=skill_repository,
                    tracer=tracing,
                )
                stack.push_async_callback(agent.aclose)
                container.chat = ChatService(
                    agent.runner,
                    repository,
                    clock,
                    MemoryPolicy(
                        context_turns=settings.memory.context_turns,
                        context_chars=settings.memory.context_chars,
                    ),
                )
                container.summarizer = SummarizeConversation(
                    agent.deps.llm,
                    repository,
                    clock,
                    every=settings.memory.summary_every_turns,
                    max_chars=settings.memory.summary_max_chars,
                )
                container.health_checks["qdrant"] = _qdrant_check(
                    agent, settings.retrieval.embedding.dimension
                )
            yield container
    finally:
        await database.dispose()
