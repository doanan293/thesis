"""The real app over a migrated Postgres database with only the services auth tests need."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx

from pharma_agent.api.app import create_app
from pharma_agent.application.conversation.queries import ConversationQueries
from pharma_agent.domain.shared.clock import SystemClock
from pharma_agent.infrastructure.container import Container
from pharma_agent.infrastructure.persistence.postgres.database import Database
from pharma_agent.infrastructure.settings import Settings
from tests.api.asgi import running
from tests.memory_repository import (
    InMemoryCitationReader,
    InMemoryConversationRepository,
    InMemoryFeedbackRepository,
)

JWT_SECRET = "s" * 40
CSRF_SECRET = "c" * 40
PASSWORD = "S3cure-password!"


@asynccontextmanager
async def postgres_app(
    dsn: str, *, auth: dict[str, object] | None = None
) -> AsyncGenerator[tuple[httpx.AsyncClient, Database]]:
    database = Database(dsn, pool_size=2)
    settings = Settings(
        _env_file=None,
        auth={"jwt_secret": JWT_SECRET, "csrf_secret": CSRF_SECRET, **(auth or {})},
        postgres={"dsn": dsn},
    )

    @asynccontextmanager
    async def factory(resolved: Settings) -> AsyncGenerator[Container]:
        yield Container(
            settings=resolved,
            sessions=database.sessions,
            queries=ConversationQueries(
                InMemoryConversationRepository(),
                SystemClock(),
                feedback=InMemoryFeedbackRepository(),
                citations=InMemoryCitationReader(),
            ),
            health_checks={"postgres": database.ping},
        )

    try:
        async with running(create_app(settings, container_factory=factory)) as client:
            yield client, database
    finally:
        await database.dispose()


def new_email() -> str:
    return f"{uuid4().hex[:12]}@example.com"


async def register(client: httpx.AsyncClient, email: str) -> None:
    response = await client.post(
        "/api/v1/auth/register", json={"email": email, "password": PASSWORD}
    )
    assert response.status_code == 201, response.text


async def cookie_login(client: httpx.AsyncClient, email: str) -> httpx.Response:
    response = await client.post(
        "/api/v1/auth/cookie/login", data={"username": email, "password": PASSWORD}
    )
    assert response.status_code == 204, response.text
    return response


def csrf_header(client: httpx.AsyncClient) -> dict[str, str]:
    return {"x-csrftoken": client.cookies["csrftoken"]}


def set_cookie(response: httpx.Response, name: str) -> str:
    """The lower-cased Set-Cookie header for `name` (a response may set several cookies)."""
    return next(
        value
        for value in response.headers.get_list("set-cookie")
        if value.startswith(f"{name}=")
    ).lower()
