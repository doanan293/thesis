"""Owns every long-lived resource of the HTTP service and wires application services."""

from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pharma_agent.application.chat.service import ChatService, MemoryPolicy
from pharma_agent.application.conversation.queries import ConversationQueries
from pharma_agent.application.memory.summarize import SummarizeConversation
from pharma_agent.domain.retrieval.ports import RetrievalError
from pharma_agent.domain.shared.clock import SystemClock
from pharma_agent.infrastructure.composition import Application, build_application
from pharma_agent.infrastructure.langgraph.checkpointer import (
    open_postgres_checkpointer,
)
from pharma_agent.infrastructure.persistence.postgres.conversation_repository import (
    AuditContext,
    PostgresConversationRepository,
)
from pharma_agent.infrastructure.persistence.postgres.database import Database
from pharma_agent.infrastructure.settings import Settings

HealthCheck = Callable[[], Awaitable[bool]]


@dataclass
class Container:
    settings: Settings
    sessions: async_sessionmaker[AsyncSession] | None
    queries: ConversationQueries
    chat: ChatService | None = None
    summarizer: SummarizeConversation | None = None
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
    container = Container(
        settings=settings,
        sessions=database.sessions,
        queries=ConversationQueries(repository, clock),
        health_checks={"postgres": database.ping},
    )
    try:
        async with AsyncExitStack() as stack:
            if settings.llm.configured:
                checkpointer = await stack.enter_async_context(
                    open_postgres_checkpointer(settings.postgres.conninfo)
                )
                agent = build_application(settings, checkpointer=checkpointer)
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
