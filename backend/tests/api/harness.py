"""Build the real FastAPI app over in-memory services and a fake authenticated user."""

import json
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import FastAPI

from pharma_agent.api.app import create_app
from pharma_agent.application.chat.graph import build_chat_graph
from pharma_agent.application.chat.runner import ChatTurnRunner
from pharma_agent.application.chat.service import ChatService, MemoryPolicy
from pharma_agent.application.conversation.queries import ConversationQueries
from pharma_agent.application.feedback.service import FeedbackService
from pharma_agent.application.memory.summarize import SummarizeConversation
from pharma_agent.application.skill.service import SkillService
from pharma_agent.domain.agent.budget import BudgetLimits
from pharma_agent.domain.shared.clock import SystemClock
from pharma_agent.infrastructure.container import Container
from pharma_agent.infrastructure.persistence.postgres.tables import UserTable
from pharma_agent.infrastructure.settings import Settings
from tests.api.asgi import running
from tests.domain.factories import make_hit
from tests.fakes import FakeLlm, FakeRetriever, build_deps
from tests.memory_repository import (
    InMemoryCitationReader,
    InMemoryConversationRepository,
    InMemoryFeedbackRepository,
    InMemorySkillRepository,
)

OWNER = uuid.UUID(hex="a" * 32)


class RecordingScoreSink:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def record_feedback(self, *, run_id: str, rating: str, note: str) -> None:
        self.calls.append((run_id, rating, note))


@dataclass
class Harness:
    app: FastAPI
    llm: FakeLlm
    repo: InMemoryConversationRepository
    container: Container
    skill_repo: InMemorySkillRepository
    feedback_repo: InMemoryFeedbackRepository
    sink: RecordingScoreSink
    citations: InMemoryCitationReader

    @asynccontextmanager
    async def client(self) -> AsyncGenerator[httpx.AsyncClient]:
        async with running(self.app) as client:
            yield client


def settings() -> Settings:
    return Settings(_env_file=None, auth={"jwt_secret": "s" * 40})


def build_harness(
    *,
    agent: bool = True,
    authenticated: bool = True,
    health: dict[str, Callable[[], Awaitable[bool]]] | None = None,
    health_reasons: dict[str, str] | None = None,
    limits: BudgetLimits | None = None,
    retriever: FakeRetriever | None = None,
) -> Harness:
    llm = FakeLlm()
    repo = InMemoryConversationRepository()
    clock = SystemClock()
    turn_retriever = (
        retriever
        if retriever is not None
        else FakeRetriever(*[[make_hit(f"c{i}", fusion=0.9)] for i in range(10)])
    )
    runner = ChatTurnRunner(
        build_chat_graph(),
        build_deps(llm, turn_retriever),
        limits if limits is not None else BudgetLimits(),
    )
    skill_repo = InMemorySkillRepository()
    feedback_repo = InMemoryFeedbackRepository()
    citation_reader = InMemoryCitationReader()
    sink = RecordingScoreSink()
    container = Container(
        settings=settings(),
        sessions=None,
        queries=ConversationQueries(
            repo, clock, feedback=feedback_repo, citations=citation_reader
        ),
        chat=ChatService(runner, repo, clock, MemoryPolicy()) if agent else None,
        summarizer=SummarizeConversation(llm, repo, clock, every=2, max_chars=500)
        if agent
        else None,
        skills=SkillService(skill_repo),
        feedback=FeedbackService(repo, feedback_repo, sink, clock),
        health_checks=health if health is not None else {"postgres": _ok},
        health_reasons=health_reasons or {},
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
    return Harness(
        app=app,
        llm=llm,
        repo=repo,
        container=container,
        skill_repo=skill_repo,
        feedback_repo=feedback_repo,
        sink=sink,
        citations=citation_reader,
    )


async def _ok() -> bool:
    return True


def parse_sse(body: str) -> list[str]:
    """The data payload of every SSE event; comment lines such as pings are skipped."""
    payloads: list[str] = []
    for block in body.replace("\r\n", "\n").split("\n\n"):
        lines = [
            line.removeprefix("data:").removeprefix(" ")
            for line in block.split("\n")
            if line.startswith("data:")
        ]
        if lines:
            payloads.append("\n".join(lines))
    return payloads


def ui_chunks(body: str) -> list[dict[str, Any]]:
    """Decoded UI Message Stream chunks; the stream must end with `data: [DONE]`."""
    payloads = parse_sse(body)
    assert payloads and payloads[-1] == "[DONE]", payloads[-1:]
    return [json.loads(payload) for payload in payloads[:-1]]
