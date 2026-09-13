"""HTTP → auth → ChatService → Postgres repository, with the fake LLM/retriever of Plan 1."""

import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from uuid import uuid4

import pytest
from sqlalchemy import func, select

from pharma_agent.api.app import create_app
from pharma_agent.application.chat.graph import build_chat_graph
from pharma_agent.application.chat.runner import ChatTurnRunner
from pharma_agent.application.chat.service import ChatService, MemoryPolicy
from pharma_agent.application.conversation.queries import ConversationQueries
from pharma_agent.application.feedback.service import FeedbackService
from pharma_agent.application.skill.service import SkillService
from pharma_agent.application.tracing import NullTracing
from pharma_agent.domain.agent.budget import BudgetLimits
from pharma_agent.domain.shared.clock import SystemClock
from pharma_agent.infrastructure.container import Container
from pharma_agent.infrastructure.persistence.postgres.citation_reader import (
    PostgresCitationReader,
)
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
from pharma_agent.infrastructure.persistence.postgres.tables import (
    FeedbackTable,
    MessageTable,
    RetrievalRunTable,
)
from pharma_agent.infrastructure.settings import Settings
from tests.api.asgi import running
from tests.api.harness import ui_chunks
from tests.api.test_chat_api import script_turn
from tests.contract.invariants import assert_stream_invariants
from tests.corpus_rows import hit_for, seed_cited_release
from tests.fakes import FakeLlm, FakeRetriever, build_deps

pytestmark = pytest.mark.integration


async def test_register_login_stream_and_persist(migrated_dsn: str) -> None:
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
            AuditContext(embedding_model="test"),
        )
        seeded = await seed_cited_release(database.sessions)
        runner = ChatTurnRunner(
            build_chat_graph(),
            build_deps(llm, FakeRetriever([hit_for(seeded)])),
            BudgetLimits(),
        )
        clock = SystemClock()
        feedback_repository = PostgresFeedbackRepository(database.sessions)
        try:
            yield Container(
                settings=resolved,
                sessions=database.sessions,
                queries=ConversationQueries(
                    repo,
                    clock,
                    feedback=feedback_repository,
                    citations=PostgresCitationReader(database.sessions),
                ),
                chat=ChatService(runner, repo, clock, MemoryPolicy()),
                skills=SkillService(PostgresSkillRepository(database.sessions)),
                feedback=FeedbackService(
                    repo, feedback_repository, NullTracing(), clock
                ),
                health_checks={"postgres": database.ping},
            )
        finally:
            await database.dispose()

    app = create_app(settings, container_factory=factory)
    email, password = f"{uuid4().hex[:10]}@example.com", "S3cure-password!"
    async with running(app) as client:
        anonymous = await client.post("/api/v1/chat", json={"message": "hi"})
        assert anonymous.status_code == 401
        assert anonymous.headers["content-type"] == "application/problem+json"
        assert anonymous.json()["code"] == "UNAUTHORIZED"
        registered = await client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": password, "display_name": "An"},
        )
        assert registered.status_code == 201, registered.text
        duplicate = await client.post(
            "/api/v1/auth/register", json={"email": email, "password": password}
        )
        assert duplicate.status_code == 400
        assert duplicate.json()["code"] == "REGISTER_USER_ALREADY_EXISTS"
        wrong_password = await client.post(
            "/api/v1/auth/jwt/login",
            data={"username": email, "password": "wrong-password"},
        )
        assert wrong_password.status_code == 400
        assert wrong_password.json()["code"] == "LOGIN_BAD_CREDENTIALS"
        login = await client.post(
            "/api/v1/auth/jwt/login", data={"username": email, "password": password}
        )
        headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
        me = await client.get("/api/v1/users/me", headers=headers)
        assert me.json()["display_name"] == "An"

        response = await client.post(
            "/api/v1/chat/stream",
            json={"message": "Paracetamol uống bao nhiêu?"},
            headers=headers,
        )
        chunks = ui_chunks(response.text)
        assert_stream_invariants(chunks)
        conversation_id = str(chunks[1]["data"]["id"])
        message_id = str(chunks[0]["messageId"])
        assert chunks[-1]["messageMetadata"]["persisted"] is True

        listed = await client.get("/api/v1/conversations", headers=headers)
        assert [item["id"] for item in listed.json()["items"]] == [conversation_id]
        messages = await client.get(
            f"/api/v1/conversations/{conversation_id}/messages", headers=headers
        )
        assert [m["role"] for m in messages.json()["items"]] == ["user", "assistant"]

        async with database_holder["db"].sessions() as session:
            message_count = (
                await session.execute(select(func.count()).select_from(MessageTable))
            ).scalar_one()
            audit_count = (
                await session.execute(
                    select(func.count()).select_from(RetrievalRunTable)
                )
            ).scalar_one()
        assert message_count >= 2 and audit_count >= 1

        skill_file = (
            "---\nname: ghi-chu-thuoc-bo\n"
            "description: Ghi chú thuốc bổ. Dùng khi hỏi về vitamin.\n---\n\n"
            "# Ghi chú thuốc bổ\n\nTrả lời ngắn gọn.\n"
        ).encode()
        uploaded = await client.post(
            "/api/v1/skills",
            headers=headers,
            files={"file": ("SKILL.md", skill_file, "text/markdown")},
        )
        assert uploaded.status_code == 201, uploaded.text
        visible = await client.get("/api/v1/skills", headers=headers)
        assert uploaded.json()["name"] in {item["name"] for item in visible.json()}

        rated = await client.post(
            f"/api/v1/messages/{message_id}/feedback",
            headers=headers,
            json={"rating": "up", "note": "đúng"},
        )
        assert rated.status_code == 201, rated.text
        async with database_holder["db"].sessions() as session:
            feedback_rows = (
                await session.execute(
                    select(func.count())
                    .select_from(FeedbackTable)
                    .where(FeedbackTable.message_id == uuid.UUID(hex=message_id))
                )
            ).scalar_one()
        assert feedback_rows == 1
