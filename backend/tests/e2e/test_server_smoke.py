"""The E2E server in-process: the exact app Playwright drives, over real Postgres and Qdrant."""

from typing import Any
from uuid import uuid4

import httpx
import pytest
from sqlalchemy.engine import make_url

from tests.api.asgi import running
from tests.api.harness import ui_chunks
from tests.api.postgres_app import PASSWORD, csrf_header, new_email
from tests.e2e.server import E2EConfig, build_e2e_app

pytestmark = pytest.mark.integration


async def stream_turn(
    client: httpx.AsyncClient, conversation_id: str, message: str
) -> list[dict[str, Any]]:
    response = await client.post(
        "/api/v1/chat/stream",
        headers=csrf_header(client),
        json={"message": message, "conversation_id": conversation_id},
    )
    assert response.status_code == 200, response.text
    assert response.headers["x-vercel-ai-ui-message-stream"] == "v1"
    return ui_chunks(response.text)


def of_type(chunks: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [chunk for chunk in chunks if chunk["type"] == kind]


async def test_register_login_chat_and_open_a_citation(
    postgres_dsn: str, qdrant_url: str
) -> None:
    dsn = (
        make_url(postgres_dsn)
        .set(database=f"e2e_{uuid4().hex[:8]}")
        .render_as_string(hide_password=False)
    )
    app = build_e2e_app(E2EConfig(postgres_dsn=dsn, qdrant_url=qdrant_url))
    email = new_email()
    async with running(app) as client:
        registered = await client.post(
            "/api/v1/auth/register", json={"email": email, "password": PASSWORD}
        )
        assert registered.status_code == 201, registered.text
        login = await client.post(
            "/api/v1/auth/cookie/login", data={"username": email, "password": PASSWORD}
        )
        assert login.status_code == 204, login.text
        me = await client.get("/api/v1/users/me")
        assert me.status_code == 200 and me.json()["email"] == email

        created = await client.post(
            "/api/v1/conversations", headers=csrf_header(client)
        )
        assert created.status_code == 201, created.text
        conversation_id = created.json()["id"]

        answered = await stream_turn(
            client, conversation_id, "Paracetamol người lớn uống bao nhiêu?"
        )
        start = answered[0]
        assert start["type"] == "start"
        assert answered[1]["data"]["id"] == conversation_id
        text = "".join(chunk["delta"] for chunk in of_type(answered, "text-delta"))
        assert "[1]" in text
        sources = of_type(answered, "source-document")
        assert [
            source["providerMetadata"]["pharma"]["index"] for source in sources
        ] == [1]
        finish = answered[-1]
        assert finish["type"] == "finish" and finish["finishReason"] == "stop"
        assert finish["messageMetadata"]["status"] == "completed"
        assert finish["messageMetadata"]["persisted"] is True

        detail = await client.get(f"/api/v1/messages/{start['messageId']}/citations/1")
        assert detail.status_code == 200, detail.text
        citation = detail.json()
        assert citation["index"] == 1 and citation["is_current"] is True
        assert [chunk["id"] for chunk in citation["chunks"] if chunk["matched"]] == [
            sources[0]["sourceId"]
        ]

        blocked = await stream_turn(
            client, conversation_id, "[e2e:blocked] Paracetamol là gì?"
        )
        assert blocked[-1]["messageMetadata"]["status"] == "blocked"
        assert of_type(blocked, "source-document") == []

        timed_out = await stream_turn(
            client, conversation_id, "[e2e:timeout] Paracetamol là gì?"
        )
        assert timed_out[-1]["finishReason"] == "error"
        assert timed_out[-1]["messageMetadata"]["status"] == "timeout"
