"""HTTP → auth → ChatService → Postgres repository, with the fake LLM/retriever of Plan 1."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from pharma_agent.api.app import create_app
from pharma_agent.application.chat.graph import build_chat_graph
from pharma_agent.application.chat.runner import ChatTurnRunner
from pharma_agent.application.chat.service import ChatService, MemoryPolicy
from pharma_agent.application.conversation.queries import ConversationQueries
from pharma_agent.domain.agent.budget import BudgetLimits
from pharma_agent.domain.shared.clock import SystemClock
from pharma_agent.infrastructure.container import Container
from pharma_agent.infrastructure.persistence.postgres.conversation_repository import (
    AuditContext,
    PostgresConversationRepository,
)
from pharma_agent.infrastructure.persistence.postgres.database import Database
from pharma_agent.infrastructure.persistence.postgres.tables import (
    MessageTable,
    RetrievalRunTable,
)
from pharma_agent.infrastructure.settings import Settings
from tests.api.harness import parse_sse
from tests.api.test_chat_api import script_turn
from tests.domain.factories import make_hit
from tests.fakes import FakeLlm, FakeRetriever, build_deps

pytestmark = pytest.mark.integration


def test_register_login_stream_and_persist(migrated_dsn: str) -> None:
    settings = Settings(
        _env_file=None, auth={"jwt_secret": "s" * 40}, postgres={"dsn": migrated_dsn}
    )
    llm = FakeLlm()
    script_turn(llm)
    database_holder: dict[str, Database] = {}

    @asynccontextmanager
    async def factory(resolved: Settings) -> AsyncGenerator[Container]:
        database = Database(resolved.postgres.dsn, pool_size=2)
        database_holder["db"] = database
        repo = PostgresConversationRepository(
            database.sessions,
            AuditContext(corpus_version="test", embedding_model="test"),
        )
        runner = ChatTurnRunner(
            build_chat_graph(),
            build_deps(llm, FakeRetriever([make_hit("c1", fusion=0.9)])),
            BudgetLimits(),
        )
        clock = SystemClock()
        try:
            yield Container(
                settings=resolved,
                sessions=database.sessions,
                queries=ConversationQueries(repo, clock),
                chat=ChatService(runner, repo, clock, MemoryPolicy()),
                health_checks={"postgres": database.ping},
            )
        finally:
            await database.dispose()

    app = create_app(settings, container_factory=factory)
    email, password = f"{uuid4().hex[:10]}@example.com", "S3cure-password!"
    with TestClient(app) as client:
        assert client.post("/api/v1/chat", json={"message": "hi"}).status_code == 401
        registered = client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": password, "display_name": "An"},
        )
        assert registered.status_code == 201, registered.text
        token = client.post(
            "/api/v1/auth/jwt/login", data={"username": email, "password": password}
        ).json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        assert (
            client.get("/api/v1/users/me", headers=headers).json()["display_name"]
            == "An"
        )

        response = client.post(
            "/api/v1/chat/stream",
            json={"message": "Paracetamol uống bao nhiêu?"},
            headers=headers,
        )
        events = parse_sse(response.text)
        conversation_id = str(events[0][1]["conversation_id"])
        assert events[-1][0] == "done" and events[-1][1]["message_id"]

        listed = client.get("/api/v1/conversations", headers=headers).json()
        assert [item["id"] for item in listed] == [conversation_id]
        messages = client.get(
            f"/api/v1/conversations/{conversation_id}/messages", headers=headers
        ).json()
        assert [m["role"] for m in messages] == ["user", "assistant"]

        async def count_rows() -> tuple[int, int]:
            async with database_holder["db"].sessions() as session:
                message_count = (
                    await session.execute(
                        select(func.count()).select_from(MessageTable)
                    )
                ).scalar_one()
                audit_count = (
                    await session.execute(
                        select(func.count()).select_from(RetrievalRunTable)
                    )
                ).scalar_one()
            return message_count, audit_count

        message_count, audit_count = client.portal.call(count_rows)
        assert message_count >= 2 and audit_count >= 1
