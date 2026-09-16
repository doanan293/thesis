"""Every error path answers with an RFC 9457 problem document (spec A §7)."""

from typing import Any

import httpx
import pytest
from fastapi import HTTPException
from fastapi_users.router.common import ErrorCode

from tests.api.harness import build_harness

PROBLEM_MEDIA_TYPE = "application/problem+json"


def assert_problem(response: httpx.Response, status: int, code: str) -> dict[str, Any]:
    assert response.status_code == status, response.text
    assert response.headers["content-type"] == PROBLEM_MEDIA_TYPE
    body = response.json()
    assert body["status"] == status
    assert body["code"] == code
    assert body["type"] == "urn:pharma-agent:problem:" + code.lower().replace("_", "-")
    assert body["title"] == code.replace("_", " ").capitalize()
    return body


async def test_application_error_is_a_problem() -> None:
    harness = build_harness()
    async with harness.client() as client:
        response = await client.get(f"/api/v1/conversations/{'f' * 32}")
    body = assert_problem(response, 404, "CONVERSATION_NOT_FOUND")
    assert body["type"] == "urn:pharma-agent:problem:conversation-not-found"
    assert body["title"] == "Conversation not found"
    assert body["detail"] == "f" * 32
    assert "errors" not in body


async def test_request_validation_error_lists_each_field() -> None:
    harness = build_harness()
    async with harness.client() as client:
        empty = await client.post("/api/v1/chat", json={"message": ""})
        bad_path = await client.get("/api/v1/conversations/nope")

    body = assert_problem(empty, 422, "VALIDATION_ERROR")
    assert "detail" not in body
    assert body["errors"] == [
        {
            "loc": ["body", "message"],
            "message": "String should have at least 1 character",
            "type": "string_too_short",
        }
    ]
    path_errors = assert_problem(bad_path, 422, "VALIDATION_ERROR")["errors"]
    assert [(item["loc"], item["type"]) for item in path_errors] == [
        (["path", "conversation_id"], "string_pattern_mismatch")
    ]


async def test_starlette_routing_errors_are_problems() -> None:
    harness = build_harness()
    async with harness.client() as client:
        missing = await client.get("/api/v1/nope")
        wrong_method = await client.delete("/api/v1/chat")

    assert "detail" not in assert_problem(missing, 404, "NOT_FOUND")
    assert_problem(wrong_method, 405, "METHOD_NOT_ALLOWED")
    assert wrong_method.headers["allow"] == "POST"


@pytest.mark.parametrize(
    ("status", "detail", "headers", "code", "expected_detail"),
    [
        (
            400,
            ErrorCode.REGISTER_USER_ALREADY_EXISTS,
            None,
            "REGISTER_USER_ALREADY_EXISTS",
            None,
        ),
        (
            400,
            {
                "code": ErrorCode.REGISTER_INVALID_PASSWORD,
                "reason": "Password is too short",
            },
            None,
            "REGISTER_INVALID_PASSWORD",
            "Password is too short",
        ),
        (400, ErrorCode.LOGIN_BAD_CREDENTIALS, None, "LOGIN_BAD_CREDENTIALS", None),
        (400, ErrorCode.OAUTH_INVALID_STATE, None, "OAUTH_INVALID_STATE", None),
        (401, None, {"WWW-Authenticate": "Bearer"}, "UNAUTHORIZED", None),
        (403, "not allowed here", None, "FORBIDDEN", "not allowed here"),
    ],
)
async def test_http_exceptions_keep_their_codes(
    status: int,
    detail: object,
    headers: dict[str, str] | None,
    code: str,
    expected_detail: str | None,
) -> None:
    harness = build_harness()

    async def fail() -> None:
        raise HTTPException(status_code=status, detail=detail, headers=headers)

    harness.app.add_api_route("/test/fail", fail, methods=["GET"])
    async with harness.client() as client:
        response = await client.get("/test/fail")

    body = assert_problem(response, status, code)
    assert body.get("detail") == expected_detail
    if headers is not None:
        assert response.headers["www-authenticate"] == "Bearer"


async def test_service_errors_have_stable_codes() -> None:
    starting = build_harness()
    transport = httpx.ASGITransport(app=starting.app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        # The lifespan has not run, so the container does not exist yet.
        assert_problem(await client.get("/api/v1/health"), 503, "SERVICE_STARTING")

    harness = build_harness(agent=False)
    harness.container.feedback = None
    async with harness.client() as client:
        chat = await client.post("/api/v1/chat", json={"message": "hi"})
        feedback = await client.post(
            f"/api/v1/messages/{'f' * 32}/feedback", json={"rating": "up"}
        )
    assert_problem(chat, 503, "AGENT_UNAVAILABLE")
    assert_problem(feedback, 503, "SERVICE_NOT_CONFIGURED")

    anonymous = build_harness(authenticated=False)
    async with anonymous.client() as client:
        # Resolving the current user needs the database, which this harness lacks.
        response = await client.get("/api/v1/conversations")
    body = assert_problem(response, 503, "SERVICE_NOT_CONFIGURED")
    assert body["detail"] == "the database is not configured"


async def test_unhandled_error_is_an_internal_error_problem() -> None:
    harness = build_harness()

    async def explode() -> None:
        raise RuntimeError("secret connection string")

    harness.app.add_api_route("/test/explode", explode, methods=["GET"])
    # Starlette re-raises after the 500 handler so the server logs the traceback;
    # the transport must not turn that into a test failure.
    transport = httpx.ASGITransport(app=harness.app, raise_app_exceptions=False)
    async with (
        harness.app.router.lifespan_context(harness.app),
        httpx.AsyncClient(transport=transport, base_url="http://testserver") as client,
    ):
        response = await client.get("/test/explode")

    body = assert_problem(response, 500, "INTERNAL_ERROR")
    assert "detail" not in body
