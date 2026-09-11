"""Build the real FastAPI app over in-memory services and a fake authenticated user."""

import json
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi.testclient import TestClient

from pharma_agent.api.app import create_app
from pharma_agent.application.chat.graph import build_chat_graph
from pharma_agent.application.chat.runner import ChatTurnRunner
from pharma_agent.application.chat.service import ChatService, MemoryPolicy
from pharma_agent.application.conversation.queries import ConversationQueries
from pharma_agent.application.memory.summarize import SummarizeConversation
from pharma_agent.domain.agent.budget import BudgetLimits
from pharma_agent.domain.shared.clock import SystemClock
from pharma_agent.infrastructure.container import Container
from pharma_agent.infrastructure.persistence.postgres.tables import UserTable
from pharma_agent.infrastructure.settings import Settings
from tests.domain.factories import make_hit
from tests.fakes import FakeLlm, FakeRetriever, build_deps
from tests.memory_repository import InMemoryConversationRepository

OWNER = uuid.UUID(hex="a" * 32)


@dataclass
class Harness:
    client: TestClient
    llm: FakeLlm
    repo: InMemoryConversationRepository
    container: Container


def settings() -> Settings:
    return Settings(_env_file=None, auth={"jwt_secret": "s" * 40})


def build_harness(
    *,
    agent: bool = True,
    authenticated: bool = True,
    health: dict[str, Callable[[], Awaitable[bool]]] | None = None,
) -> Harness:
    llm = FakeLlm()
    repo = InMemoryConversationRepository()
    clock = SystemClock()
    retriever = FakeRetriever(*[[make_hit(f"c{i}", fusion=0.9)] for i in range(10)])
    runner = ChatTurnRunner(
        build_chat_graph(), build_deps(llm, retriever), BudgetLimits()
    )
    container = Container(
        settings=settings(),
        sessions=None,
        queries=ConversationQueries(repo, clock),
        chat=ChatService(runner, repo, clock, MemoryPolicy()) if agent else None,
        summarizer=SummarizeConversation(llm, repo, clock, every=2, max_chars=500)
        if agent
        else None,
        health_checks=health if health is not None else {"postgres": _ok},
    )

    @asynccontextmanager
    async def factory(_: Settings) -> AsyncGenerator[Container]:
        yield container

    app = create_app(settings(), container_factory=factory)
    if authenticated:
        auth = app.state.auth
        app.dependency_overrides[auth.current_active_user] = lambda: UserTable(
            id=OWNER, email="owner@example.com", hashed_password="x", is_active=True
        )
    return Harness(client=TestClient(app), llm=llm, repo=repo, container=container)


async def _ok() -> bool:
    return True


def parse_sse(body: str) -> list[tuple[str, dict[str, object]]]:
    events: list[tuple[str, dict[str, object]]] = []
    for block in body.replace("\r\n", "\n").split("\n\n"):
        name, data = "", ""
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data += line.removeprefix("data:").strip()
        if name and data:
            events.append((name, json.loads(data)))
    return events
