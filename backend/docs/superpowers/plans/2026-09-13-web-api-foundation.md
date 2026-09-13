# Web API Foundation Implementation Plan (Plan 5 of 10)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the web client a stable HTTP contract: every error is an RFC 9457 problem document with a stable `code`, conversation and message lists page with opaque keyset cursors, conversations can be created before the first message, and the OpenAPI document has clean operation ids and can be exported without a server.

**Architecture:** Application errors gain a `ServiceUnavailable` family and `pharma_agent.application.pagination` (cursor encode/decode). The domain `ConversationRepository` port switches from `before: datetime` to a `(timestamp, id)` keyset cursor, and `Conversation` gains `create_empty` and `title_from_first_message`. The API layer maps every exception (application errors, `HTTPException` including fastapi-users codes, `RequestValidationError`, Starlette 404/405, unhandled errors) to `application/problem+json` through FastAPI exception handlers. A `FastAPI` subclass post-processes the generated OpenAPI document so every error response points at `Problem`. The `pharma-agent export-openapi` command writes that document deterministically.

**Tech Stack:** FastAPI 0.141 exception handlers and `generate_unique_id_function`, Starlette 1.6, fastapi-users 15.0.5 (`ErrorCode` details), Pydantic 2.13, SQLAlchemy 2.0 row-value comparison (`tuple_`), Alembic, Typer, pytest with `httpx.ASGITransport` (`tests/api/asgi.py`), testcontainers Postgres (`tests/postgres.py`).

**Spec:** `backend/docs/superpowers/specs/2026-09-13-web-client-contract-design.md` §4.1 (conversation endpoints, cursors), §7 (RFC 9457 errors), §8 (OpenAPI), §10 rows "Phân trang", "Lỗi", "OpenAPI". Builds on P3 (migration `0006` is the Alembic head before this plan). P6 later replaces `MessagePage.items` (`MessageView` here) with `UIMessage`; P7 adds cookie auth routes and `CSRF_FAILED`.

## Global Constraints

- Development environment only: Postgres may be reset; no backfill, no backward compatibility for the old list shapes (`list[...]` bodies, `before` query parameter, `{"code","message"}` errors).
- Python 3.12, uv, shared `ruff.toml`, strict `pyrefly.toml` with `unused-ignore = true`, pytest `filterwarnings = ["error"]`. Fix lint and type errors in code; never add rule ignores, `# noqa`, `# type: ignore` or new `# pyrefly: ignore`.
- Every task ends green on: `uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q`. Tasks touching Postgres also run `uv run pytest -q -m integration`. All commands run from `backend/`.
- Layering stays enforced by `tests/architecture/test_layering.py`: the domain imports no framework and no outer layer; `api/` never imports `pharma_agent.domain`.
- Prefer established libraries; no feature flag or "fake mode" in production code; fakes live under `tests/`.
- API tests go through `tests/api/harness.py` and `tests/api/asgi.py` (`httpx.ASGITransport`), never `starlette.testclient`. Integration tests use `pytestmark = pytest.mark.integration` and the `migrated_dsn` fixture.
- One commit per task, conventional message, ending with the session attribution trailer of the executing session.
- Pinned names (overview §3.5): operation ids equal route function names (`generate_unique_id_function=lambda route: route.name`); response models `ConversationView`, `ConversationPage`, `MessagePage`, `FeedbackView`, `SkillView`, `HealthResponse`, `Problem`, `ProblemItem`; error `type` is `urn:pharma-agent:problem:<code in lower kebab case>`; `pharma_agent.application.pagination.encode_cursor(timestamp: datetime, id: str) -> str` and `decode_cursor(cursor: str) -> tuple[datetime, str]` raising `InvalidCursor` (code `INVALID_CURSOR`); CLI `pharma-agent export-openapi --output PATH`; Alembic revision `0007`.
- Plan-specific exact values:
  - Problem media type `application/problem+json`. Body keys: `type`, `title`, `status`, `detail` (omitted when empty), `code`, `errors` (only on 422 `VALIDATION_ERROR`). `title` is the code humanised: `code.replace("_", " ").capitalize()` (`CONVERSATION_NOT_FOUND` → `Conversation not found`).
  - `ProblemItem = {loc: list[str | int], message: str, type: str}`.
  - Codes produced here: `VALIDATION_ERROR` (422), `INVALID_CURSOR` (422), `SERVICE_STARTING` (503, container not ready), `SERVICE_NOT_CONFIGURED` (503, database/skills/feedback missing), `AGENT_UNAVAILABLE` (503), `INTERNAL_ERROR` (500), the existing application codes, fastapi-users `ErrorCode` values taken from `detail` (string or `{"code","reason"}`), and for any other `HTTPException` the `http.HTTPStatus(status).name` (`UNAUTHORIZED`, `FORBIDDEN`, `NOT_FOUND`, `METHOD_NOT_ALLOWED`, `BAD_REQUEST`). `HTTPException.headers` (for example `Allow`, `WWW-Authenticate`) are preserved.
  - Cursor: unpadded base64url of compact JSON `{"t": "<ISO-8601 with offset>", "id": "<32-hex id>"}`; decoding accepts padded or unpadded input and hyphenated or hex UUIDs, and returns the id as 32 lowercase hex characters.
  - `GET /conversations?limit=20&cursor=`: `limit` 1..100, default 20; order `(updated_at desc, id desc)`; only `turn_count > 0`.
  - `GET /conversations/{id}/messages?limit=30&cursor=`: `limit` 1..100, default 30; first page is the newest messages; items oldest → newest inside a page; `next_cursor` points to older messages.
  - `cursor` query parameter: `max_length=512`.
  - Migration `0007` (down revision `0006`) replaces `ix_conversations_user_updated` with `ix_conversations_user_updated_id (user_id, updated_at, id)` and `ix_messages_conversation_created` with `ix_messages_conversation_created_id (conversation_id, created_at, id)`.
  - Default conversation title constant `DEFAULT_TITLE = "Cuộc trò chuyện mới"` in `pharma_agent.domain.conversation.models`.
  - OpenAPI export settings are fixed in code (JWT secret placeholder, Google OAuth enabled with placeholder credentials) so the document always contains the OAuth routes and never depends on `.env`.
  - Operation ids after this plan (P6 adds `get_message_citation`; P7 adds `auth:cookie.login`, `auth:cookie.logout` and renames the OAuth ids to `oauth:google.cookie.*`): `health`, `chat`, `chat_stream`, `create_conversation`, `list_conversations`, `get_conversation`, `list_messages`, `rename_conversation`, `delete_conversation`, `submit_feedback`, `list_skills`, `upload_skill`, `set_skill_enabled`, `delete_skill`, `auth:jwt.login`, `auth:jwt.logout`, `register:register`, `users:current_user`, `users:patch_current_user`, `users:user`, `users:patch_user`, `users:delete_user`, `oauth:google.jwt.authorize`, `oauth:google.jwt.callback`.

---

## File Structure

```text
backend/
  src/pharma_agent/
    application/errors.py                      (+ ServiceUnavailable, ServiceStarting, ServiceNotConfigured; AgentUnavailable joins the family)  # Task 1
    application/pagination.py                  encode_cursor, decode_cursor, InvalidCursor                                                     # Task 2
    domain/conversation/ports.py               list_for_user/messages take a (timestamp, id) keyset cursor                                      # Task 3
    domain/conversation/models.py              DEFAULT_TITLE, Conversation.create_empty, Conversation.title_from_first_message                 # Task 4
    infrastructure/persistence/postgres/tables.py                     keyset indexes                                                           # Task 3
    infrastructure/persistence/postgres/conversation_repository.py    keyset queries, turn_count > 0 filter                                    # Task 3
    infrastructure/persistence/postgres/migrations/versions/0007_keyset_indexes.py                                                             # Task 3
    application/conversation/queries.py        ConversationPage, MessagePage, paged list queries (Task 3); create (Task 4)
    application/chat/service.py                name a pre-created conversation after its first message                                          # Task 4
    api/schemas.py                             Problem, ProblemItem (ErrorResponse removed)                                                     # Task 1
    api/problems.py                            problem_type, humanize_code, ProblemResponse, problem_response (Task 1); problem_responses (Task 5)
    api/errors.py                              exception handlers for every error path                                                          # Task 1
    api/deps.py                                503s raise ServiceStarting / ServiceNotConfigured                                                # Task 1
    api/routers/conversations.py               paged lists (Task 3), POST /conversations (Task 4), problem responses (Task 5)
    api/routers/{chat,feedback,skills,health}.py                      problem responses in OpenAPI                                             # Task 5
    api/openapi.py                             PharmaAgentAPI (OpenAPI post-processing), openapi_export_settings, render_openapi                # Task 5
    api/app.py                                 PharmaAgentAPI with generate_unique_id_function                                                  # Task 5
    cli.py                                     export-openapi                                                                                   # Task 5
  tests/
    api/test_problem_details.py                problem+json for each error group                                                                # Task 1
    api/test_skills_api.py                     reads `detail` instead of `message`                                                              # Task 1
    api/test_e2e_postgres.py                   401 / fastapi-users codes (Task 1), page shapes (Task 3)
    application/test_pagination.py             cursor round trip and rejection                                                                   # Task 2
    memory_repository.py                       in-memory keyset behaviour identical to Postgres                                                 # Task 3
    infrastructure/test_conversation_repository.py                    keyset pages with identical timestamps                                   # Task 3
    infrastructure/test_migrations.py          indexes of 0007                                                                                  # Task 3
    application/test_queries.py                ConversationPage / MessagePage paging (Task 3), create (Task 4)
    infrastructure/test_container.py           page shape                                                                                       # Task 3
    api/test_conversations_api.py              paged lists and INVALID_CURSOR (Task 3), POST /conversations (Task 4)
    domain/test_conversation.py                create_empty, title_from_first_message                                                           # Task 4
    application/test_chat_service.py           first turn names a pre-created conversation                                                     # Task 4
    api/test_openapi.py                        operation ids, problem responses, components                                                      # Task 5
    test_cli.py                                export-openapi                                                                                   # Task 5
```

---

### Task 1: RFC 9457 problem details for every error path

**Files:**
- Modify: `backend/src/pharma_agent/application/errors.py` (whole file, lines 1-18)
- Modify: `backend/src/pharma_agent/api/schemas.py` (lines 24-26: `ErrorResponse` is replaced by `ProblemItem` and `Problem`)
- Create: `backend/src/pharma_agent/api/problems.py`
- Modify: `backend/src/pharma_agent/api/errors.py` (whole file, lines 1-38)
- Modify: `backend/src/pharma_agent/api/deps.py` (lines 1-52)
- Create: `backend/tests/api/test_problem_details.py`
- Modify: `backend/tests/api/test_skills_api.py` (line 88)
- Modify: `backend/tests/api/test_e2e_postgres.py` (lines 89-99)

**Interfaces:**
- Consumes: `ApplicationError.code` and its subclasses (`ConversationNotFound`, `InvalidInput`, `PayloadTooLarge`, `MessageNotFound`, `SkillNotFound`, `SkillNameTaken`); `starlette.exceptions.HTTPException` (`status_code`, `detail`, `headers`), which `fastapi.HTTPException` and `httpx_oauth`'s `OAuth2AuthorizeCallbackError` subclass; `fastapi_users.router.common.ErrorCode` (a `str, Enum`) raised as `detail=ErrorCode.X` or `detail={"code": ErrorCode.X, "reason": str}`; `RequestValidationError.errors()` items `{"loc": tuple, "msg": str, "type": str, ...}`.
- Produces:
  - `pharma_agent.application.errors`: `class ServiceUnavailable(ApplicationError)` code `SERVICE_UNAVAILABLE`; `class ServiceStarting(ServiceUnavailable)` code `SERVICE_STARTING`; `class ServiceNotConfigured(ServiceUnavailable)` code `SERVICE_NOT_CONFIGURED`; `AgentUnavailable` now subclasses `ServiceUnavailable` (code unchanged).
  - `pharma_agent.api.schemas`: `class ProblemItem(BaseModel): loc: list[str | int]; message: str; type: str`; `class Problem(BaseModel): type: str; title: str; status: int; detail: str | None = None; code: str; errors: list[ProblemItem] | None = None`.
  - `pharma_agent.api.problems`: `PROBLEM_MEDIA_TYPE = "application/problem+json"`, `problem_type(code: str) -> str`, `humanize_code(code: str) -> str`, `class ProblemResponse(JSONResponse)`, `problem_response(status: int, code: str, *, detail: str | None = None, errors: list[ProblemItem] | None = None, headers: Mapping[str, str] | None = None) -> ProblemResponse`.
  - `pharma_agent.api.errors`: `STATUS_BY_ERROR`, `VALIDATION_ERROR = "VALIDATION_ERROR"`, `INTERNAL_ERROR = "INTERNAL_ERROR"`, handlers `application_error_handler`, `http_exception_handler`, `validation_error_handler`, `unhandled_error_handler` (all `async (Request, Exception) -> Response`), `install_error_handlers(app: FastAPI) -> None`.

- [ ] **Step 1: Write the failing tests**

`backend/tests/api/test_problem_details.py`:

```python
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
    harness.container.skills = None
    harness.container.feedback = None
    async with harness.client() as client:
        chat = await client.post("/api/v1/chat", json={"message": "hi"})
        skills = await client.get("/api/v1/skills")
        feedback = await client.post(
            f"/api/v1/messages/{'f' * 32}/feedback", json={"rating": "up"}
        )
    assert_problem(chat, 503, "AGENT_UNAVAILABLE")
    assert_problem(skills, 503, "SERVICE_NOT_CONFIGURED")
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
```

In `backend/tests/api/test_skills_api.py` replace line 88:

```python
        assert "lowercase" in invalid_name.json()["detail"]
```

In `backend/tests/api/test_e2e_postgres.py` replace lines 89-99 (from `async with running(app) as client:` through the `login = await client.post(...)` call) with:

```python
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
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/api/test_problem_details.py tests/api/test_skills_api.py -q`
Expected: FAIL. Responses still use `application/json` with `{"code","message"}` or `{"detail": ...}`, so `assert response.headers["content-type"] == PROBLEM_MEDIA_TYPE` fails; `test_service_errors_have_stable_codes` gets `{"detail": "service is starting"}`; `test_skills_api.py::test_upload_errors` fails with `KeyError: 'detail'`.

- [ ] **Step 3: Add the service-unavailable errors**

`backend/src/pharma_agent/application/errors.py`:

```python
class ApplicationError(Exception):
    code = "APPLICATION_ERROR"


class ConversationNotFound(ApplicationError):
    code = "CONVERSATION_NOT_FOUND"


class InvalidInput(ApplicationError):
    code = "INVALID_INPUT"


class PayloadTooLarge(InvalidInput):
    code = "PAYLOAD_TOO_LARGE"


class ServiceUnavailable(ApplicationError):
    """Something the request needs is not ready or not configured (HTTP 503)."""

    code = "SERVICE_UNAVAILABLE"


class ServiceStarting(ServiceUnavailable):
    code = "SERVICE_STARTING"


class ServiceNotConfigured(ServiceUnavailable):
    code = "SERVICE_NOT_CONFIGURED"


class AgentUnavailable(ServiceUnavailable):
    code = "AGENT_UNAVAILABLE"
```

- [ ] **Step 4: Add the problem models and response helpers**

In `backend/src/pharma_agent/api/schemas.py` replace the `ErrorResponse` class (lines 24-26) with:

```python
class ProblemItem(BaseModel):
    """One invalid request field of a 422 problem."""

    loc: list[str | int]
    message: str
    type: str


class Problem(BaseModel):
    """RFC 9457 problem details; clients switch on `code`."""

    type: str
    title: str
    status: int
    detail: str | None = None
    code: str
    errors: list[ProblemItem] | None = None
```

`backend/src/pharma_agent/api/problems.py`:

```python
"""RFC 9457 problem details: the single response shape of every API error."""

from collections.abc import Mapping

from fastapi.responses import JSONResponse

from pharma_agent.api.schemas import Problem, ProblemItem

PROBLEM_MEDIA_TYPE = "application/problem+json"
PROBLEM_TYPE_PREFIX = "urn:pharma-agent:problem:"


def problem_type(code: str) -> str:
    """`CONVERSATION_NOT_FOUND` -> `urn:pharma-agent:problem:conversation-not-found`."""
    return PROBLEM_TYPE_PREFIX + code.lower().replace("_", "-")


def humanize_code(code: str) -> str:
    """`CONVERSATION_NOT_FOUND` -> `Conversation not found`."""
    return code.replace("_", " ").capitalize()


class ProblemResponse(JSONResponse):
    media_type = PROBLEM_MEDIA_TYPE


def problem_response(
    status: int,
    code: str,
    *,
    detail: str | None = None,
    errors: list[ProblemItem] | None = None,
    headers: Mapping[str, str] | None = None,
) -> ProblemResponse:
    problem = Problem(
        type=problem_type(code),
        title=humanize_code(code),
        status=status,
        detail=detail or None,
        code=code,
        errors=errors,
    )
    return ProblemResponse(
        problem.model_dump(mode="json", exclude_none=True),
        status_code=status,
        headers=headers,
    )
```

- [ ] **Step 5: Map every exception to a problem**

`backend/src/pharma_agent/api/errors.py`:

```python
"""FastAPI exception handlers: every error leaves the API as application/problem+json."""

import re
from enum import Enum
from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from starlette.responses import Response

from pharma_agent.api.problems import problem_response
from pharma_agent.api.schemas import ProblemItem
from pharma_agent.application.errors import (
    ApplicationError,
    ConversationNotFound,
    InvalidInput,
    PayloadTooLarge,
    ServiceUnavailable,
)
from pharma_agent.application.feedback.service import MessageNotFound
from pharma_agent.application.skill.service import SkillNameTaken, SkillNotFound

VALIDATION_ERROR = "VALIDATION_ERROR"
INTERNAL_ERROR = "INTERNAL_ERROR"
HTTP_ERROR = "HTTP_ERROR"

# fastapi-users raises HTTPException with an upper snake case ErrorCode in `detail`.
ERROR_CODE_PATTERN = re.compile(r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*")

# Subclasses must come before their base class: the first isinstance match wins.
STATUS_BY_ERROR: dict[type[ApplicationError], int] = {
    ConversationNotFound: 404,
    MessageNotFound: 404,
    SkillNotFound: 404,
    SkillNameTaken: 409,
    PayloadTooLarge: 413,
    InvalidInput: 422,
    ServiceUnavailable: 503,
}


def _status_name(status: int) -> str:
    try:
        return HTTPStatus(status).name
    except ValueError:
        return HTTP_ERROR


def _status_phrase(status: int) -> str:
    try:
        return HTTPStatus(status).phrase
    except ValueError:
        return ""


def _plain(value: object) -> object:
    return value.value if isinstance(value, Enum) else value


def _code_and_detail(exc: HTTPException) -> tuple[str, str | None]:
    fallback = _status_name(exc.status_code)
    detail = _plain(exc.detail)
    if isinstance(detail, dict):
        code, reason = _plain(detail.get("code")), detail.get("reason")
        valid = isinstance(code, str) and ERROR_CODE_PATTERN.fullmatch(code)
        return (
            code if valid and isinstance(code, str) else fallback,
            reason if isinstance(reason, str) else None,
        )
    if isinstance(detail, str):
        if ERROR_CODE_PATTERN.fullmatch(detail):
            return detail, None
        if detail != _status_phrase(exc.status_code):
            return fallback, detail
    return fallback, None


async def application_error_handler(request: Request, exc: Exception) -> Response:
    if not isinstance(exc, ApplicationError):
        raise exc
    status = next(
        (code for kind, code in STATUS_BY_ERROR.items() if isinstance(exc, kind)), 400
    )
    return problem_response(status, exc.code, detail=str(exc))


async def http_exception_handler(request: Request, exc: Exception) -> Response:
    if not isinstance(exc, HTTPException):
        raise exc
    code, detail = _code_and_detail(exc)
    return problem_response(
        exc.status_code, code, detail=detail, headers=exc.headers
    )


async def validation_error_handler(request: Request, exc: Exception) -> Response:
    if not isinstance(exc, RequestValidationError):
        raise exc
    errors = [
        ProblemItem(
            loc=list(error["loc"]), message=str(error["msg"]), type=str(error["type"])
        )
        for error in exc.errors()
    ]
    return problem_response(422, VALIDATION_ERROR, errors=errors)


async def unhandled_error_handler(request: Request, exc: Exception) -> Response:
    # Never echo the exception text: it may contain internals. Starlette re-raises
    # the exception after this response so the server still logs the traceback.
    return problem_response(500, INTERNAL_ERROR)


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ApplicationError, application_error_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)
```

Registering the handler on `starlette.exceptions.HTTPException` replaces FastAPI's default JSON handler for both Starlette's routing errors (404, 405) and every `fastapi.HTTPException`, including the fastapi-users and httpx-oauth ones.

- [ ] **Step 6: Raise application errors for 503s in the dependencies**

In `backend/src/pharma_agent/api/deps.py` replace lines 1-52 (imports through `require_feedback`) with:

```python
from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pharma_agent.application.chat.service import ChatService
from pharma_agent.application.errors import (
    AgentUnavailable,
    ServiceNotConfigured,
    ServiceStarting,
)
from pharma_agent.application.feedback.service import FeedbackService
from pharma_agent.application.skill.service import SkillService
from pharma_agent.infrastructure.auth.users import Auth
from pharma_agent.infrastructure.container import Container
from pharma_agent.infrastructure.persistence.postgres.tables import UserTable

UserIdDependency = Callable[..., Awaitable[str]]


def get_container(request: Request) -> Container:
    container = getattr(request.app.state, "container", None)
    if not isinstance(container, Container):
        raise ServiceStarting("the service is starting")
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


def session_factory(request: Request) -> async_sessionmaker[AsyncSession]:
    sessions = get_container(request).sessions
    if sessions is None:
        raise ServiceNotConfigured("the database is not configured")
    return sessions


def require_chat(container: Container) -> ChatService:
    if container.chat is None:
        raise AgentUnavailable(
            "the agent is not configured (set PHARMA_LLM__DEFAULT__API_KEY)"
        )
    return container.chat


def require_skills(container: Container) -> SkillService:
    if container.skills is None:
        raise ServiceNotConfigured("skills are not configured")
    return container.skills


def require_feedback(container: Container) -> FeedbackService:
    if container.feedback is None:
        raise ServiceNotConfigured("feedback is not configured")
    return container.feedback
```

`user_id_dependency` (lines 55-61) stays unchanged. `install_error_handlers(app)` is already called by `create_app`.

- [ ] **Step 7: Run the tests to see them pass**

Run: `uv run pytest tests/api -q`
Expected: PASS (all API tests, including the unchanged `test_chat_api.py`, `test_feedback_api.py` and `test_skills_api.py` code assertions).

- [ ] **Step 8: Run the full check**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q && uv run pytest -q -m integration`
Expected: all green; `tests/api/test_e2e_postgres.py` proves the real fastapi-users 401, `REGISTER_USER_ALREADY_EXISTS` and `LOGIN_BAD_CREDENTIALS` responses are problems.

- [ ] **Step 9: Commit**

```bash
git add backend/src/pharma_agent/application/errors.py backend/src/pharma_agent/api/schemas.py backend/src/pharma_agent/api/problems.py backend/src/pharma_agent/api/errors.py backend/src/pharma_agent/api/deps.py backend/tests/api/test_problem_details.py backend/tests/api/test_skills_api.py backend/tests/api/test_e2e_postgres.py
git commit -m "feat(api): return RFC 9457 problem details for every error"
```

The commit message ends with the session attribution trailer.

---

### Task 2: Keyset cursor helpers

**Files:**
- Create: `backend/src/pharma_agent/application/pagination.py`
- Create: `backend/tests/application/test_pagination.py`

**Interfaces:**
- Consumes: `pharma_agent.application.errors.InvalidInput` (mapped to 422 by `STATUS_BY_ERROR` from Task 1).
- Produces: `class InvalidCursor(InvalidInput)` with `code = "INVALID_CURSOR"`; `encode_cursor(timestamp: datetime, id: str) -> str`; `decode_cursor(cursor: str) -> tuple[datetime, str]` (UTC timestamp, 32-hex id).

- [ ] **Step 1: Write the failing tests**

`backend/tests/application/test_pagination.py`:

```python
import base64
import json
from datetime import UTC, datetime, timedelta, timezone

import pytest

from pharma_agent.application.errors import InvalidInput
from pharma_agent.application.pagination import (
    InvalidCursor,
    decode_cursor,
    encode_cursor,
)

AT = datetime(2026, 9, 13, 8, 12, 0, 123456, tzinfo=UTC)
ROW_ID = "3f2b6c1e9a7d4b8c8e0f1a2b3c4d5e6f"


def b64(value: object) -> str:
    return base64.urlsafe_b64encode(json.dumps(value).encode()).decode()


def payload_of(cursor: str) -> object:
    return json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))


def test_round_trip_keeps_microseconds_and_id() -> None:
    cursor = encode_cursor(AT, ROW_ID)
    assert not set(cursor) & {"=", "+", "/"}
    assert decode_cursor(cursor) == (AT, ROW_ID)
    assert payload_of(cursor) == {"t": "2026-09-13T08:12:00.123456+00:00", "id": ROW_ID}


def test_other_offsets_are_stored_as_utc() -> None:
    hanoi = AT.astimezone(timezone(timedelta(hours=7)))
    cursor = encode_cursor(hanoi, ROW_ID)
    assert payload_of(cursor) == {"t": "2026-09-13T08:12:00.123456+00:00", "id": ROW_ID}
    decoded, _ = decode_cursor(cursor)
    assert decoded.utcoffset() == timedelta(0)


def test_decode_accepts_padding_and_hyphenated_uuid() -> None:
    cursor = b64({"t": AT.isoformat(), "id": "12345678-1234-5678-1234-567812345678"})
    assert cursor.endswith("=")
    assert decode_cursor(cursor) == (AT, "12345678123456781234567812345678")


def test_encode_rejects_naive_timestamps() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        encode_cursor(datetime(2026, 9, 13, 8, 12), ROW_ID)


@pytest.mark.parametrize(
    "cursor",
    [
        "",
        "!!!",
        "é",
        "bm90IGpzb24",
        b64(["t", "id"]),
        b64({"t": AT.isoformat()}),
        b64({"t": 1, "id": ROW_ID}),
        b64({"t": "yesterday", "id": ROW_ID}),
        b64({"t": "2026-09-13T08:12:00", "id": ROW_ID}),
        b64({"t": AT.isoformat(), "id": "not-a-uuid"}),
    ],
)
def test_malformed_cursors_are_rejected(cursor: str) -> None:
    with pytest.raises(InvalidCursor) as raised:
        decode_cursor(cursor)
    assert raised.value.code == "INVALID_CURSOR"
    assert isinstance(raised.value, InvalidInput)
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/application/test_pagination.py -q`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'pharma_agent.application.pagination'`.

- [ ] **Step 3: Implement the helpers**

`backend/src/pharma_agent/application/pagination.py`:

```python
"""Opaque keyset cursors for paged lists (spec A §4.1).

A cursor is the unpadded base64url encoding of the compact JSON
`{"t": "<ISO-8601 timestamp>", "id": "<row id>"}` of the last row a page returned;
the next page continues strictly after that `(timestamp, id)` position.
"""

import base64
import json
import uuid
from datetime import UTC, datetime

from pharma_agent.application.errors import InvalidInput

INVALID_CURSOR_MESSAGE = "the cursor is not valid"


class InvalidCursor(InvalidInput):
    code = "INVALID_CURSOR"


def encode_cursor(timestamp: datetime, id: str) -> str:
    if timestamp.tzinfo is None:
        raise ValueError("cursor timestamps must be timezone-aware")
    payload = json.dumps(
        {"t": timestamp.astimezone(UTC).isoformat(), "id": id},
        separators=(",", ":"),
    )
    encoded = base64.urlsafe_b64encode(payload.encode("utf-8"))
    return encoded.rstrip(b"=").decode("ascii")


def decode_cursor(cursor: str) -> tuple[datetime, str]:
    # binascii.Error, UnicodeDecodeError and JSONDecodeError are all ValueErrors.
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
        payload = json.loads(raw)
    except ValueError as exc:
        raise InvalidCursor(INVALID_CURSOR_MESSAGE) from exc
    if not isinstance(payload, dict):
        raise InvalidCursor(INVALID_CURSOR_MESSAGE)
    timestamp, row_id = payload.get("t"), payload.get("id")
    if not isinstance(timestamp, str) or not isinstance(row_id, str):
        raise InvalidCursor(INVALID_CURSOR_MESSAGE)
    try:
        at = datetime.fromisoformat(timestamp)
        key = uuid.UUID(row_id)
    except ValueError as exc:
        raise InvalidCursor(INVALID_CURSOR_MESSAGE) from exc
    if at.tzinfo is None:
        raise InvalidCursor(INVALID_CURSOR_MESSAGE)
    return at.astimezone(UTC), key.hex
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `uv run pytest tests/application/test_pagination.py -q`
Expected: PASS (14 tests).

- [ ] **Step 5: Run the full check**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add backend/src/pharma_agent/application/pagination.py backend/tests/application/test_pagination.py
git commit -m "feat(application): add opaque keyset cursor helpers"
```

The commit message ends with the session attribution trailer.

---

### Task 3: Keyset pagination for conversations and messages

Changing the repository port breaks its callers, so the port, both repositories, the migration, the queries and the routes change together in this one deliverable.

**Files:**
- Modify: `backend/src/pharma_agent/domain/conversation/ports.py` (`list_for_user` lines 16-20, `messages` lines 50-54)
- Modify: `backend/src/pharma_agent/infrastructure/persistence/postgres/tables.py` (`ConversationTable.__table_args__` line 74, `MessageTable.__table_args__` lines 98-101)
- Create: `backend/src/pharma_agent/infrastructure/persistence/postgres/migrations/versions/0007_keyset_indexes.py`
- Modify: `backend/src/pharma_agent/infrastructure/persistence/postgres/conversation_repository.py` (import line 7, `list_for_user` lines 124-136, `messages` lines 225-237)
- Modify: `backend/tests/memory_repository.py` (`list_for_user` lines 36-44, `messages` lines 89-97)
- Modify: `backend/src/pharma_agent/application/conversation/queries.py` (imports, new page models after `MessageView`, `list_conversations` lines 61-67, `list_messages` lines 74-86)
- Modify: `backend/src/pharma_agent/api/routers/conversations.py` (whole file)
- Modify: `backend/tests/infrastructure/test_conversation_repository.py` (`test_append_turn_is_atomic_and_increments_turn_count` lines 169-172, `test_list_for_user_orders_by_recent_activity_and_deletes_cascade` lines 227-253, new test appended)
- Modify: `backend/tests/infrastructure/test_migrations.py` (new test appended)
- Modify: `backend/tests/application/test_queries.py` (whole file)
- Modify: `backend/tests/api/test_conversations_api.py` (whole file)
- Modify: `backend/tests/infrastructure/test_container.py` (line 20)
- Modify: `backend/tests/api/test_e2e_postgres.py` (the `listed` and `messages` assertions after the stream)

Line numbers refer to commit `8f76523`; P1-P3 may have shifted them, so locate the named functions.

**Interfaces:**
- Consumes: `encode_cursor`, `decode_cursor`, `InvalidCursor` (Task 2); problem handlers (Task 1).
- Produces:
  - `ConversationRepository.list_for_user(self, user_id: str, *, limit: int, cursor: tuple[datetime, str] | None = None) -> list[Conversation]`: only `turn_count > 0`, ordered `(updated_at desc, id desc)`, strictly after the cursor.
  - `ConversationRepository.messages(self, conversation_id: str, *, limit: int, cursor: tuple[datetime, str] | None = None) -> list[Message]`: the newest `limit` messages strictly older than the `(created_at, id)` cursor, returned oldest first.
  - `pharma_agent.application.conversation.queries.ConversationPage(items: list[ConversationView], next_cursor: str | None)`, `MessagePage(items: list[MessageView], next_cursor: str | None)`.
  - `ConversationQueries.list_conversations(self, user_id: str, *, limit: int, cursor: str | None = None) -> ConversationPage`; `ConversationQueries.list_messages(self, user_id: str, conversation_id: str, *, limit: int, cursor: str | None = None) -> MessagePage`. An empty `cursor` string means the first page.
  - Routes `list_conversations` (`GET /conversations`, `limit` 1..100 default 20, `cursor`) and `list_messages` (`GET /conversations/{conversation_id}/messages`, `limit` 1..100 default 30, `cursor`), responses `ConversationPage` and `MessagePage`.
  - Alembic revision `0007` (down revision `0006`).

- [ ] **Step 1: Write the failing tests**

`backend/tests/application/test_queries.py` (whole file):

```python
from datetime import timedelta

import pytest

from pharma_agent.application.conversation.queries import ConversationQueries
from pharma_agent.application.errors import ConversationNotFound, InvalidInput
from pharma_agent.application.pagination import InvalidCursor
from pharma_agent.domain.conversation.models import Conversation
from pharma_agent.domain.conversation.turns import build_turn_messages
from pharma_agent.domain.shared.clock import FixedClock
from tests.domain.factories import make_run
from tests.fakes import NOW
from tests.memory_repository import InMemoryConversationRepository

OWNER, STRANGER = "a" * 32, "b" * 32


async def add_turn(
    repo: InMemoryConversationRepository, conversation: Conversation
) -> None:
    """Append a turn stamped NOW, so timestamps tie across turns."""
    user_msg, assistant_msg = build_turn_messages(
        conversation_id=conversation.conversation_id,
        run=make_run("q"),
        answer_text="a",
        citations=[],
        phases=[],
        now=NOW,
    )
    conversation.record_turn(NOW)
    await repo.append_turn(conversation, user_msg, assistant_msg, [])


async def test_list_get_messages_rename_delete() -> None:
    repo = InMemoryConversationRepository()
    queries = ConversationQueries(repo, FixedClock(NOW + timedelta(days=1)))
    older = Conversation.start(user_id=OWNER, first_message="older", now=NOW)
    never_used = Conversation.start(
        user_id=OWNER, first_message="never used", now=NOW + timedelta(minutes=1)
    )
    await repo.create(older)
    await repo.create(never_used)
    user_msg, assistant_msg = build_turn_messages(
        conversation_id=older.conversation_id,
        run=make_run("q"),
        answer_text="a",
        citations=[],
        phases=["answering"],
        now=NOW,
    )
    older.record_turn(NOW + timedelta(minutes=2))
    await repo.append_turn(older, user_msg, assistant_msg, [])

    listed = await queries.list_conversations(OWNER, limit=10)
    assert [c.title for c in listed.items] == ["older"]
    assert listed.items[0].turn_count == 1 and listed.next_cursor is None

    page = await queries.list_messages(OWNER, older.conversation_id, limit=10)
    assert [(m.role, m.content) for m in page.items] == [
        ("user", "q"),
        ("assistant", "a"),
    ]
    assert page.items[1].phases == ["answering"] and page.next_cursor is None

    renamed = await queries.rename(OWNER, older.conversation_id, "  Thuốc hạ sốt ")
    assert renamed.title == "Thuốc hạ sốt" and renamed.updated_at == NOW + timedelta(
        days=1
    )

    with pytest.raises(InvalidInput):
        await queries.rename(OWNER, older.conversation_id, " ")
    for call in (
        queries.get_conversation(STRANGER, older.conversation_id),
        queries.list_messages(STRANGER, older.conversation_id, limit=10),
        queries.rename(STRANGER, older.conversation_id, "x"),
        queries.delete(STRANGER, older.conversation_id),
    ):
        with pytest.raises(ConversationNotFound):
            await call

    await queries.delete(OWNER, older.conversation_id)
    assert (await queries.list_conversations(OWNER, limit=10)).items == []


async def test_pages_have_no_duplicates_or_gaps_when_timestamps_tie() -> None:
    repo = InMemoryConversationRepository()
    queries = ConversationQueries(repo, FixedClock(NOW))
    created: list[Conversation] = []
    for index in range(5):
        conversation = Conversation.start(
            user_id=OWNER, first_message=f"c{index}", now=NOW
        )
        await repo.create(conversation)
        await add_turn(repo, conversation)
        created.append(conversation)
    for _ in range(3):
        await add_turn(repo, created[0])
    await repo.create(Conversation.start(user_id=OWNER, first_message="empty", now=NOW))

    listed: list[str] = []
    cursor: str | None = None
    while True:
        conversations = await queries.list_conversations(OWNER, limit=2, cursor=cursor)
        listed.extend(item.id for item in conversations.items)
        if conversations.next_cursor is None:
            break
        cursor = conversations.next_cursor
    assert listed == sorted((c.conversation_id for c in created), reverse=True)

    conversation_id = created[0].conversation_id
    pages: list[list[str]] = []
    cursor = None
    while True:
        messages = await queries.list_messages(
            OWNER, conversation_id, limit=3, cursor=cursor
        )
        pages.append([m.id for m in messages.items])
        if messages.next_cursor is None:
            break
        cursor = messages.next_cursor
    everything = [m.message_id for m in await repo.messages(conversation_id, limit=100)]
    assert len(set(everything)) == 8
    assert [len(ids) for ids in pages] == [3, 3, 2]
    assert [mid for ids in reversed(pages) for mid in ids] == everything

    exact = await queries.list_messages(OWNER, conversation_id, limit=8)
    assert len(exact.items) == 8 and exact.next_cursor is None
    first_page = await queries.list_conversations(OWNER, limit=2, cursor="")
    assert len(first_page.items) == 2
    with pytest.raises(InvalidCursor):
        await queries.list_conversations(OWNER, limit=2, cursor="nope")
```

`backend/tests/api/test_conversations_api.py` (whole file):

```python
from datetime import UTC, datetime, timedelta

from pharma_agent.domain.agent.run import AnswerMode, AnswerPlan
from pharma_agent.domain.conversation.models import Conversation
from pharma_agent.domain.conversation.turns import build_turn_messages
from tests.api.harness import OWNER, Harness, build_harness
from tests.domain.factories import make_run


async def add_turn(harness: Harness, conversation: Conversation, at: datetime) -> None:
    run = make_run("q")
    run.submit_plan(AnswerPlan(mode=AnswerMode.NO_RETRIEVAL), now=at)
    run.complete()
    user_msg, assistant_msg = build_turn_messages(
        conversation_id=conversation.conversation_id,
        run=run,
        answer_text="a",
        citations=[],
        phases=[],
        now=at,
    )
    conversation.record_turn(at)
    await harness.repo.append_turn(conversation, user_msg, assistant_msg, [])


async def test_conversation_endpoints() -> None:
    harness = build_harness()
    now = datetime.now(UTC)
    mine = Conversation.start(user_id=OWNER.hex, first_message="Paracetamol", now=now)
    foreign = Conversation.start(
        user_id="b" * 32, first_message="Không phải của tôi", now=now
    )
    await harness.repo.create(mine)
    await harness.repo.create(foreign)
    await add_turn(harness, mine, now + timedelta(seconds=1))

    async with harness.client() as client:
        listed = await client.get("/api/v1/conversations")
        assert listed.status_code == 200
        assert [item["title"] for item in listed.json()["items"]] == ["Paracetamol"]
        assert listed.json()["next_cursor"] is None

        detail = await client.get(f"/api/v1/conversations/{mine.conversation_id}")
        assert detail.json()["turn_count"] == 1

        messages = (
            await client.get(f"/api/v1/conversations/{mine.conversation_id}/messages")
        ).json()
        assert [m["role"] for m in messages["items"]] == ["user", "assistant"]

        renamed = await client.patch(
            f"/api/v1/conversations/{mine.conversation_id}",
            json={"title": "Thuốc hạ sốt"},
        )
        assert renamed.status_code == 200 and renamed.json()["title"] == "Thuốc hạ sốt"
        blank = await client.patch(
            f"/api/v1/conversations/{mine.conversation_id}", json={"title": ""}
        )
        assert blank.status_code == 422

        foreign_get = await client.get(
            f"/api/v1/conversations/{foreign.conversation_id}"
        )
        assert foreign_get.status_code == 404
        foreign_delete = await client.delete(
            f"/api/v1/conversations/{foreign.conversation_id}"
        )
        assert foreign_delete.status_code == 404
        deleted = await client.delete(f"/api/v1/conversations/{mine.conversation_id}")
        assert deleted.status_code == 204
        assert (await client.get("/api/v1/conversations")).json() == {
            "items": [],
            "next_cursor": None,
        }


async def test_lists_page_with_cursors() -> None:
    harness = build_harness()
    now = datetime.now(UTC)
    created: list[Conversation] = []
    for index in range(3):
        conversation = Conversation.start(
            user_id=OWNER.hex, first_message=f"c{index}", now=now
        )
        await harness.repo.create(conversation)
        await add_turn(harness, conversation, now)
        created.append(conversation)
    for _ in range(2):
        await add_turn(harness, created[0], now)
    empty = Conversation.start(user_id=OWNER.hex, first_message="empty", now=now)
    await harness.repo.create(empty)
    target = created[0].conversation_id

    async with harness.client() as client:
        first = (await client.get("/api/v1/conversations", params={"limit": 2})).json()
        assert len(first["items"]) == 2 and first["next_cursor"]
        second = (
            await client.get(
                "/api/v1/conversations",
                params={"limit": 2, "cursor": first["next_cursor"]},
            )
        ).json()
        assert second["next_cursor"] is None
        listed = [item["id"] for item in first["items"] + second["items"]]
        assert listed == sorted((c.conversation_id for c in created), reverse=True)
        assert empty.conversation_id not in listed

        url = f"/api/v1/conversations/{target}/messages"
        newest = (await client.get(url, params={"limit": 4})).json()
        older = (
            await client.get(
                url, params={"limit": 4, "cursor": newest["next_cursor"]}
            )
        ).json()
        assert [len(newest["items"]), len(older["items"])] == [4, 2]
        assert older["next_cursor"] is None
        timeline = older["items"] + newest["items"]
        assert len({m["id"] for m in timeline}) == 6
        stamps = [m["created_at"] for m in timeline]
        assert stamps == sorted(stamps)


async def test_list_parameters_are_validated() -> None:
    harness = build_harness()
    conversation = Conversation.start(
        user_id=OWNER.hex, first_message="x", now=datetime.now(UTC)
    )
    await harness.repo.create(conversation)
    async with harness.client() as client:
        bad = await client.get("/api/v1/conversations", params={"cursor": "nope"})
        bad_messages = await client.get(
            f"/api/v1/conversations/{conversation.conversation_id}/messages",
            params={"cursor": "e30"},  # base64url of "{}"
        )
        too_many = await client.get("/api/v1/conversations", params={"limit": 101})
        too_long = await client.get(
            "/api/v1/conversations", params={"cursor": "a" * 513}
        )
    for response in (bad, bad_messages):
        assert response.status_code == 422
        assert response.json()["code"] == "INVALID_CURSOR"
    for response in (too_many, too_long):
        assert response.status_code == 422
        assert response.json()["code"] == "VALIDATION_ERROR"
```

In `backend/tests/infrastructure/test_conversation_repository.py`:

1. Add `from datetime import datetime, timedelta` in place of `from datetime import timedelta` (line 3).
2. In `test_append_turn_is_atomic_and_increments_turn_count` replace the `older = await repo.messages(...)` call (lines 169-171) with:

```python
    older = await repo.messages(
        conversation.conversation_id,
        limit=10,
        cursor=(messages[0].created_at, messages[0].message_id),
    )
```

3. Replace `test_list_for_user_orders_by_recent_activity_and_deletes_cascade` (lines 227-253) with:

```python
async def test_list_for_user_orders_by_recent_activity_and_deletes_cascade(
    database: Database,
) -> None:
    repo = repository(database)
    owner = await make_user(database)
    first = Conversation.start(user_id=owner, first_message="first", now=NOW)
    second = Conversation.start(
        user_id=owner, first_message="second", now=NOW + timedelta(minutes=5)
    )
    empty = Conversation.start(
        user_id=owner, first_message="empty", now=NOW + timedelta(minutes=30)
    )
    for conversation in (first, second, empty):
        await repo.create(conversation)
    for conversation, index in ((second, 7), (first, 10)):
        user_msg, assistant_msg = turn(conversation.conversation_id, index)
        conversation.record_turn(assistant_msg.created_at)
        await repo.append_turn(
            conversation, user_msg, assistant_msg, AUDIT if conversation is first else []
        )

    listed = await repo.list_for_user(owner, limit=10)
    assert [c.title for c in listed] == ["first", "second"]
    after_first = (listed[0].updated_at, listed[0].conversation_id)
    assert [
        c.title for c in await repo.list_for_user(owner, limit=10, cursor=after_first)
    ] == ["second"]

    assert await repo.delete(owner, first.conversation_id) is True
    async with database.sessions() as session:
        assert (
            await session.execute(select(func.count()).select_from(RetrievalHitTable))
        ).scalar_one() == 0
```

4. Append:

```python
def tied_turn(conversation_id: str) -> tuple[Message, Message]:
    """Both messages share NOW, so only the id breaks the tie."""
    return (
        Message(
            message_id=uuid.uuid4().hex,
            conversation_id=conversation_id,
            role=MessageRole.USER,
            content="q",
            status="completed",
            created_at=NOW,
        ),
        Message(
            message_id=uuid.uuid4().hex,
            conversation_id=conversation_id,
            role=MessageRole.ASSISTANT,
            content="a",
            status="completed",
            created_at=NOW,
        ),
    )


async def test_keyset_pages_have_no_duplicates_or_gaps_with_tied_timestamps(
    database: Database,
) -> None:
    repo = repository(database)
    owner = await make_user(database)
    created: list[Conversation] = []
    for index in range(5):
        conversation = Conversation.start(
            user_id=owner, first_message=f"c{index}", now=NOW
        )
        await repo.create(conversation)
        user_msg, assistant_msg = tied_turn(conversation.conversation_id)
        conversation.record_turn(NOW)
        await repo.append_turn(conversation, user_msg, assistant_msg, [])
        created.append(conversation)
    await repo.create(
        Conversation.start(
            user_id=owner, first_message="empty", now=NOW + timedelta(hours=1)
        )
    )

    seen: list[str] = []
    position: tuple[datetime, str] | None = None
    while page := await repo.list_for_user(owner, limit=2, cursor=position):
        seen.extend(c.conversation_id for c in page)
        position = (page[-1].updated_at, page[-1].conversation_id)
    assert seen == sorted((c.conversation_id for c in created), reverse=True)

    target = created[0]
    for _ in range(3):
        user_msg, assistant_msg = tied_turn(target.conversation_id)
        target.record_turn(NOW)
        await repo.append_turn(target, user_msg, assistant_msg, [])
    everything = [
        m.message_id for m in await repo.messages(target.conversation_id, limit=100)
    ]
    assert everything == sorted(everything) and len(set(everything)) == 8

    pages: list[list[str]] = []
    position = None
    while page := await repo.messages(target.conversation_id, limit=3, cursor=position):
        ids = [m.message_id for m in page]
        assert ids == sorted(ids)
        pages.append(ids)
        position = (page[0].created_at, page[0].message_id)
    assert [len(ids) for ids in pages] == [3, 3, 2]
    assert [mid for ids in reversed(pages) for mid in ids] == everything
```

In `backend/tests/infrastructure/test_migrations.py` append:

```python
def _indexes(connection: Connection, table: str) -> dict[str | None, list[str | None]]:
    return {
        index["name"]: list(index["column_names"])
        for index in inspect(connection).get_indexes(table)
    }


async def test_keyset_indexes_cover_the_sort_keys(migrated_dsn: str) -> None:
    engine = create_async_engine(migrated_dsn)
    async with engine.connect() as connection:
        conversations = await connection.run_sync(_indexes, "conversations")
        messages = await connection.run_sync(_indexes, "messages")
    await engine.dispose()
    assert conversations["ix_conversations_user_updated_id"] == [
        "user_id",
        "updated_at",
        "id",
    ]
    assert "ix_conversations_user_updated" not in conversations
    assert messages["ix_messages_conversation_created_id"] == [
        "conversation_id",
        "created_at",
        "id",
    ]
    assert "ix_messages_conversation_created" not in messages
```

In `backend/tests/infrastructure/test_container.py` replace line 20 with:

```python
        page = await container.queries.list_conversations("a" * 32, limit=5)
        assert page.items == [] and page.next_cursor is None
```

In `backend/tests/api/test_e2e_postgres.py` replace the two list assertions after the stream with:

```python
        listed = await client.get("/api/v1/conversations", headers=headers)
        assert [item["id"] for item in listed.json()["items"]] == [conversation_id]
        messages = await client.get(
            f"/api/v1/conversations/{conversation_id}/messages", headers=headers
        )
        assert [m["role"] for m in messages.json()["items"]] == ["user", "assistant"]
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/application/test_queries.py tests/api/test_conversations_api.py -q`
Expected: FAIL: `AttributeError: 'list' object has no attribute 'items'` in `test_queries.py`, `TypeError: ... got an unexpected keyword argument 'cursor'` in the paging test, and `TypeError: list indices must be integers or slices, not str` in the API tests.

Run: `uv run pytest -q -m integration tests/infrastructure/test_conversation_repository.py tests/infrastructure/test_migrations.py`
Expected: FAIL: `TypeError: ... unexpected keyword argument 'cursor'` and `KeyError: 'ix_conversations_user_updated_id'`.

- [ ] **Step 3: Change the repository port**

In `backend/src/pharma_agent/domain/conversation/ports.py` replace `list_for_user` with:

```python
    async def list_for_user(
        self,
        user_id: str,
        *,
        limit: int,
        cursor: tuple[datetime, str] | None = None,
    ) -> list[Conversation]:
        """Conversations with at least one turn, ordered (updated_at desc, id desc).

        With a cursor, only the rows strictly after that (updated_at, id) position.
        """
        ...
```

and `messages` with:

```python
    async def messages(
        self,
        conversation_id: str,
        *,
        limit: int,
        cursor: tuple[datetime, str] | None = None,
    ) -> list[Message]:
        """The newest `limit` messages strictly older than the (created_at, id) cursor.

        Returned oldest first; ties on created_at are broken by id.
        """
        ...
```

- [ ] **Step 4: Add the indexes and migration 0007**

In `backend/src/pharma_agent/infrastructure/persistence/postgres/tables.py`, `ConversationTable`:

```python
    __table_args__ = (
        Index("ix_conversations_user_updated_id", "user_id", "updated_at", "id"),
    )
```

and `MessageTable`:

```python
    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant')", name="role"),
        Index(
            "ix_messages_conversation_created_id",
            "conversation_id",
            "created_at",
            "id",
        ),
    )
```

`backend/src/pharma_agent/infrastructure/persistence/postgres/migrations/versions/0007_keyset_indexes.py`:

```python
"""Keyset pagination indexes: the sort keys end with the id tie-breaker.

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-13 12:00:00
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_index("ix_conversations_user_updated", table_name="conversations")
    op.create_index(
        "ix_conversations_user_updated_id",
        "conversations",
        ["user_id", "updated_at", "id"],
        unique=False,
    )
    op.drop_index("ix_messages_conversation_created", table_name="messages")
    op.create_index(
        "ix_messages_conversation_created_id",
        "messages",
        ["conversation_id", "created_at", "id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_messages_conversation_created_id", table_name="messages")
    op.create_index(
        "ix_messages_conversation_created",
        "messages",
        ["conversation_id", "created_at"],
        unique=False,
    )
    op.drop_index("ix_conversations_user_updated_id", table_name="conversations")
    op.create_index(
        "ix_conversations_user_updated",
        "conversations",
        ["user_id", "updated_at"],
        unique=False,
    )
```

- [ ] **Step 5: Keyset queries in the Postgres repository**

In `backend/src/pharma_agent/infrastructure/persistence/postgres/conversation_repository.py` change the SQLAlchemy import to `from sqlalchemy import delete, select, tuple_, update` and replace `list_for_user`:

```python
    async def list_for_user(
        self,
        user_id: str,
        *,
        limit: int,
        cursor: tuple[datetime, str] | None = None,
    ) -> list[Conversation]:
        owner = _uuid(user_id)
        if owner is None:
            return []
        query = select(ConversationTable).where(
            ConversationTable.user_id == owner, ConversationTable.turn_count > 0
        )
        if cursor is not None:
            updated_at, key = cursor
            after = _uuid(key)
            if after is None:
                return []
            # Row-value comparison matches the index (user_id, updated_at, id).
            query = query.where(
                tuple_(ConversationTable.updated_at, ConversationTable.id)
                < tuple_(updated_at, after)
            )
        query = query.order_by(
            ConversationTable.updated_at.desc(), ConversationTable.id.desc()
        ).limit(limit)
        async with self._sessions() as session:
            rows = (await session.execute(query)).scalars().all()
        return [_conversation(row) for row in rows]
```

and `messages`:

```python
    async def messages(
        self,
        conversation_id: str,
        *,
        limit: int,
        cursor: tuple[datetime, str] | None = None,
    ) -> list[Message]:
        key = _uuid(conversation_id)
        if key is None:
            return []
        query = select(MessageTable).where(MessageTable.conversation_id == key)
        if cursor is not None:
            created_at, message_key = cursor
            before = _uuid(message_key)
            if before is None:
                return []
            query = query.where(
                tuple_(MessageTable.created_at, MessageTable.id)
                < tuple_(created_at, before)
            )
        query = query.order_by(
            MessageTable.created_at.desc(), MessageTable.id.desc()
        ).limit(limit)
        async with self._sessions() as session:
            rows = (await session.execute(query)).scalars().all()
        return [_message(row) for row in reversed(rows)]
```

`recent_turns` keeps calling `self.messages(conversation_id, limit=limit * 2)`.

- [ ] **Step 6: Same behaviour in the in-memory repository**

In `backend/tests/memory_repository.py` replace `list_for_user`:

```python
    async def list_for_user(
        self,
        user_id: str,
        *,
        limit: int,
        cursor: tuple[datetime, str] | None = None,
    ) -> list[Conversation]:
        # Hex ids compare like Postgres uuids (bytewise), so the order matches.
        rows = sorted(
            (
                row
                for row in self.rows.values()
                if row.user_id == user_id and row.turn_count > 0
            ),
            key=lambda row: (row.updated_at, row.conversation_id),
            reverse=True,
        )
        if cursor is not None:
            rows = [
                row for row in rows if (row.updated_at, row.conversation_id) < cursor
            ]
        return [row.model_copy(deep=True) for row in rows[:limit]]
```

and `messages`:

```python
    async def messages(
        self,
        conversation_id: str,
        *,
        limit: int,
        cursor: tuple[datetime, str] | None = None,
    ) -> list[Message]:
        ordered = sorted(
            self.message_log.get(conversation_id, []),
            key=lambda message: (message.created_at, message.message_id),
        )
        if cursor is not None:
            ordered = [
                message
                for message in ordered
                if (message.created_at, message.message_id) < cursor
            ]
        return ordered[-limit:]
```

- [ ] **Step 7: Page models and paged queries**

In `backend/src/pharma_agent/application/conversation/queries.py` add the import `from pharma_agent.application.pagination import decode_cursor, encode_cursor`, add after `MessageView`:

```python
class ConversationPage(BaseModel):
    items: list[ConversationView]
    next_cursor: str | None


class MessagePage(BaseModel):
    """One page of history: items oldest first, `next_cursor` points to older ones."""

    items: list[MessageView]
    next_cursor: str | None
```

and replace `list_conversations` and `list_messages` with:

```python
    async def list_conversations(
        self, user_id: str, *, limit: int, cursor: str | None = None
    ) -> ConversationPage:
        after = decode_cursor(cursor) if cursor else None
        rows = await self._conversations.list_for_user(
            user_id, limit=limit + 1, cursor=after
        )
        items = rows[:limit]
        next_cursor = (
            encode_cursor(items[-1].updated_at, items[-1].conversation_id)
            if len(rows) > limit
            else None
        )
        return ConversationPage(
            items=[ConversationView.of(row) for row in items], next_cursor=next_cursor
        )

    async def list_messages(
        self,
        user_id: str,
        conversation_id: str,
        *,
        limit: int,
        cursor: str | None = None,
    ) -> MessagePage:
        before = decode_cursor(cursor) if cursor else None
        await self._owned(user_id, conversation_id)
        rows = await self._conversations.messages(
            conversation_id, limit=limit + 1, cursor=before
        )
        # Rows are oldest first; the extra row means older messages remain.
        has_older = len(rows) > limit
        items = rows[1:] if has_older else rows
        next_cursor = (
            encode_cursor(items[0].created_at, items[0].message_id)
            if has_older
            else None
        )
        return MessagePage(
            items=[MessageView.of(row) for row in items], next_cursor=next_cursor
        )
```

The `from datetime import datetime` import stays (used by the views).

- [ ] **Step 8: Paged routes**

`backend/src/pharma_agent/api/routers/conversations.py`:

```python
from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query, Response

from pharma_agent.api.deps import ContainerDep, UserIdDependency
from pharma_agent.api.schemas import CONVERSATION_ID_PATTERN, RenameConversationRequest
from pharma_agent.application.conversation.queries import (
    ConversationPage,
    ConversationView,
    MessagePage,
)

MAX_PAGE_SIZE = 100
MAX_CURSOR_LENGTH = 512

ConversationId = Annotated[str, Path(pattern=CONVERSATION_ID_PATTERN)]
Cursor = Annotated[
    str | None,
    Query(
        max_length=MAX_CURSOR_LENGTH,
        description="`next_cursor` of the previous page; omit for the first page",
    ),
]


def build_conversations_router(current_user_id: UserIdDependency) -> APIRouter:
    router = APIRouter(prefix="/conversations", tags=["conversations"])
    UserId = Annotated[str, Depends(current_user_id)]

    @router.get("", response_model=ConversationPage)
    async def list_conversations(
        user_id: UserId,
        container: ContainerDep,
        limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 20,
        cursor: Cursor = None,
    ) -> ConversationPage:
        return await container.queries.list_conversations(
            user_id, limit=limit, cursor=cursor
        )

    @router.get("/{conversation_id}", response_model=ConversationView)
    async def get_conversation(
        conversation_id: ConversationId, user_id: UserId, container: ContainerDep
    ) -> ConversationView:
        return await container.queries.get_conversation(user_id, conversation_id)

    @router.get("/{conversation_id}/messages", response_model=MessagePage)
    async def list_messages(
        conversation_id: ConversationId,
        user_id: UserId,
        container: ContainerDep,
        limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 30,
        cursor: Cursor = None,
    ) -> MessagePage:
        return await container.queries.list_messages(
            user_id, conversation_id, limit=limit, cursor=cursor
        )

    @router.patch("/{conversation_id}", response_model=ConversationView)
    async def rename_conversation(
        conversation_id: ConversationId,
        body: RenameConversationRequest,
        user_id: UserId,
        container: ContainerDep,
    ) -> ConversationView:
        return await container.queries.rename(user_id, conversation_id, body.title)

    @router.delete("/{conversation_id}", status_code=204)
    async def delete_conversation(
        conversation_id: ConversationId, user_id: UserId, container: ContainerDep
    ) -> Response:
        await container.queries.delete(user_id, conversation_id)
        return Response(status_code=204)

    return router
```

- [ ] **Step 9: Run the tests to see them pass**

Run: `uv run pytest tests/application/test_queries.py tests/api/test_conversations_api.py -q && uv run pytest -q -m integration tests/infrastructure/test_conversation_repository.py tests/infrastructure/test_migrations.py tests/infrastructure/test_container.py tests/api/test_e2e_postgres.py`
Expected: PASS. `test_upgrade_head_matches_the_models` reports no diff, so `tables.py` and `0007` agree; `test_downgrade_then_upgrade_round_trips` exercises the downgrade.

- [ ] **Step 10: Run the full check**

Run: `uv run ruff format src tests && uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q && uv run pytest -q -m integration`
Expected: all green.

- [ ] **Step 11: Commit**

```bash
git add backend/src/pharma_agent/domain/conversation/ports.py backend/src/pharma_agent/infrastructure/persistence/postgres/tables.py backend/src/pharma_agent/infrastructure/persistence/postgres/migrations/versions/0007_keyset_indexes.py backend/src/pharma_agent/infrastructure/persistence/postgres/conversation_repository.py backend/src/pharma_agent/application/conversation/queries.py backend/src/pharma_agent/api/routers/conversations.py backend/tests/memory_repository.py backend/tests/application/test_queries.py backend/tests/api/test_conversations_api.py backend/tests/infrastructure/test_conversation_repository.py backend/tests/infrastructure/test_migrations.py backend/tests/infrastructure/test_container.py backend/tests/api/test_e2e_postgres.py
git commit -m "feat(api): page conversations and messages with keyset cursors"
```

The commit message ends with the session attribution trailer.

---

### Task 4: `POST /conversations` and naming from the first message

**Files:**
- Modify: `backend/src/pharma_agent/domain/conversation/models.py` (lines 57-97: `MAX_TITLE_CHARS` through `Conversation.start`)
- Modify: `backend/src/pharma_agent/application/conversation/queries.py` (new `create` method after `get_conversation`)
- Modify: `backend/src/pharma_agent/application/chat/service.py` (`ChatService.open_turn`, the `else:` branch, lines 159-173)
- Modify: `backend/src/pharma_agent/api/routers/conversations.py` (new route before `list_conversations`)
- Modify: `backend/tests/domain/test_conversation.py` (import line 5-10, new tests appended)
- Modify: `backend/tests/application/test_queries.py` (new test appended)
- Modify: `backend/tests/application/test_chat_service.py` (import, new tests appended)
- Modify: `backend/tests/api/test_conversations_api.py` (import, new test appended)

**Interfaces:**
- Consumes: `ConversationRepository.create`, `ConversationRepository.update_title` (persists `title` and `updated_at`), `ConversationQueries` pages (Task 3).
- Produces:
  - `pharma_agent.domain.conversation.models.DEFAULT_TITLE = "Cuộc trò chuyện mới"`.
  - `Conversation.create_empty(cls, *, user_id: str, now: datetime, conversation_id: str | None = None) -> Conversation`.
  - `Conversation.title_from_first_message(self, message: str) -> bool`: sets the title from the message only when `turn_count == 0`, the title is still `DEFAULT_TITLE` and the cleaned message is not blank; returns whether it changed; `updated_at` is untouched.
  - `Conversation.start(...)` keeps its signature and behaviour, now `create_empty` + `title_from_first_message`.
  - `ConversationQueries.create(self, user_id: str) -> ConversationView`.
  - Route `create_conversation`: `POST /conversations`, no body, 201 `ConversationView`.
  - `ChatService.open_turn` persists the new title through `update_title` before the turn starts, so the `conversation` event already carries it.

- [ ] **Step 1: Write the failing tests**

In `backend/tests/domain/test_conversation.py` add `DEFAULT_TITLE` to the `pharma_agent.domain.conversation.models` import and append:

```python
def test_create_empty_uses_the_default_title() -> None:
    conversation = Conversation.create_empty(user_id="u1", now=NOW)
    assert conversation.title == DEFAULT_TITLE == "Cuộc trò chuyện mới"
    assert conversation.turn_count == 0
    assert conversation.created_at == conversation.updated_at == NOW
    assert len(conversation.conversation_id) == 32
    assert Conversation.start(user_id="u1", first_message="   ", now=NOW).title == (
        DEFAULT_TITLE
    )


def test_first_message_names_only_an_untouched_conversation() -> None:
    fresh = Conversation.create_empty(user_id="u1", now=NOW)
    later = NOW + timedelta(minutes=1)
    assert fresh.title_from_first_message("  Paracetamol   uống bao nhiêu? ") is True
    assert fresh.title == "Paracetamol uống bao nhiêu?" and fresh.updated_at == NOW
    assert fresh.title_from_first_message("Còn trẻ em?") is False

    long_message = Conversation.create_empty(user_id="u1", now=NOW)
    assert long_message.title_from_first_message("x" * 200) is True
    assert len(long_message.title) == 80

    blank = Conversation.create_empty(user_id="u1", now=NOW)
    assert blank.title_from_first_message("   ") is False
    assert blank.title == DEFAULT_TITLE

    renamed = Conversation.create_empty(user_id="u1", now=NOW)
    renamed.rename("Thuốc hạ sốt", now=later)
    assert renamed.title_from_first_message("Paracetamol?") is False
    assert renamed.title == "Thuốc hạ sốt"

    answered = Conversation.create_empty(user_id="u1", now=NOW)
    answered.record_turn(later)
    assert answered.title_from_first_message("Paracetamol?") is False
    assert answered.title == DEFAULT_TITLE
```

In `backend/tests/application/test_queries.py` append:

```python
async def test_create_starts_an_empty_unlisted_conversation() -> None:
    repo = InMemoryConversationRepository()
    queries = ConversationQueries(repo, FixedClock(NOW))

    view = await queries.create(OWNER)

    assert (view.title, view.turn_count, view.created_at, view.updated_at) == (
        "Cuộc trò chuyện mới",
        0,
        NOW,
        NOW,
    )
    assert repo.rows[view.id].user_id == OWNER
    assert (await queries.get_conversation(OWNER, view.id)).id == view.id
    assert (await queries.list_conversations(OWNER, limit=10)).items == []
```

In `backend/tests/application/test_chat_service.py` add `from pharma_agent.domain.conversation.models import Conversation` and append:

```python
async def test_first_turn_names_a_pre_created_conversation() -> None:
    llm, repo = FakeLlm(), InMemoryConversationRepository()
    scripted_turn(llm, "Liều paracetamol cho người lớn")
    scripted_turn(llm, "Paracetamol có dùng cho trẻ em không")
    service = service_with(llm, repo)
    empty = Conversation.create_empty(user_id=OWNER, now=NOW)
    await repo.create(empty)

    session = await service.open_turn(
        user_id=OWNER,
        message="Paracetamol uống bao nhiêu?",
        conversation_id=empty.conversation_id,
    )
    events = await collect(session.events())

    assert events[0].data == {
        "conversation_id": empty.conversation_id,
        "title": "Paracetamol uống bao nhiêu?",
        "created": False,
    }
    stored = repo.rows[empty.conversation_id]
    assert stored.title == "Paracetamol uống bao nhiêu?" and stored.turn_count == 1

    await service.ask(
        user_id=OWNER,
        message="Còn trẻ em thì sao?",
        conversation_id=empty.conversation_id,
    )
    assert repo.rows[empty.conversation_id].title == "Paracetamol uống bao nhiêu?"


async def test_renamed_empty_conversation_keeps_its_title() -> None:
    llm, repo = FakeLlm(), InMemoryConversationRepository()
    scripted_turn(llm, "Liều paracetamol")
    service = service_with(llm, repo)
    empty = Conversation.create_empty(user_id=OWNER, now=NOW)
    empty.rename("Hỏi về thuốc hạ sốt", now=NOW)
    await repo.create(empty)

    await service.ask(
        user_id=OWNER, message="Paracetamol?", conversation_id=empty.conversation_id
    )

    assert repo.rows[empty.conversation_id].title == "Hỏi về thuốc hạ sốt"
```

In `backend/tests/api/test_conversations_api.py` add `from tests.api.test_chat_api import script_turn` and append:

```python
async def test_create_conversation_then_first_message_names_it() -> None:
    harness = build_harness()
    script_turn(harness.llm)
    async with harness.client() as client:
        created = await client.post("/api/v1/conversations")
        assert created.status_code == 201, created.text
        body = created.json()
        assert set(body) == {"id", "title", "turn_count", "created_at", "updated_at"}
        assert body["title"] == "Cuộc trò chuyện mới" and body["turn_count"] == 0
        conversation_id = body["id"]
        assert harness.repo.rows[conversation_id].user_id == OWNER.hex
        assert (await client.get("/api/v1/conversations")).json()["items"] == []

        chat = await client.post(
            "/api/v1/chat",
            json={
                "message": "Paracetamol uống bao nhiêu?",
                "conversation_id": conversation_id,
            },
        )
        assert chat.status_code == 200, chat.text
        assert chat.json()["conversation_id"] == conversation_id

        listed = (await client.get("/api/v1/conversations")).json()["items"]
        assert [(i["id"], i["title"], i["turn_count"]) for i in listed] == [
            (conversation_id, "Paracetamol uống bao nhiêu?", 1)
        ]
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/domain/test_conversation.py tests/application/test_queries.py tests/application/test_chat_service.py tests/api/test_conversations_api.py -q`
Expected: FAIL: `ImportError: cannot import name 'DEFAULT_TITLE'` in the domain test, `AttributeError: type object 'Conversation' has no attribute 'create_empty'` in the application tests, `AttributeError: 'ConversationQueries' object has no attribute 'create'`, and `assert 405 == 201` for `POST /api/v1/conversations`.

- [ ] **Step 3: Domain: empty conversations and the first-message title**

In `backend/src/pharma_agent/domain/conversation/models.py` replace lines 57-97 (`MAX_TITLE_CHARS` through the end of `Conversation.start`) with:

```python
MAX_TITLE_CHARS = 80
DEFAULT_TITLE = "Cuộc trò chuyện mới"


class InvalidTitle(DomainError):
    code = "INVALID_TITLE"


def _clean_title(text: str) -> str:
    return " ".join(text.split())


class Conversation(BaseModel):
    conversation_id: str
    user_id: str
    title: str
    summary: str = ""
    turn_count: int = 0
    summarized_turns: int = 0
    created_at: datetime
    updated_at: datetime

    @classmethod
    def create_empty(
        cls, *, user_id: str, now: datetime, conversation_id: str | None = None
    ) -> "Conversation":
        return cls(
            conversation_id=conversation_id or new_id(),
            user_id=user_id,
            title=DEFAULT_TITLE,
            created_at=now,
            updated_at=now,
        )

    @classmethod
    def start(
        cls,
        *,
        user_id: str,
        first_message: str,
        now: datetime,
        conversation_id: str | None = None,
    ) -> "Conversation":
        conversation = cls.create_empty(
            user_id=user_id, now=now, conversation_id=conversation_id
        )
        conversation.title_from_first_message(first_message)
        return conversation

    def title_from_first_message(self, message: str) -> bool:
        """Name a conversation nobody has named or used yet after its first question.

        Returns whether the title changed. `updated_at` is left to the turn itself.
        """
        if self.turn_count > 0 or self.title != DEFAULT_TITLE:
            return False
        title = _clean_title(message)[:MAX_TITLE_CHARS].rstrip()
        if not title or title == self.title:
            return False
        self.title = title
        return True
```

The remaining methods (`record_turn`, `needs_summary`, `apply_summary`, `rename`) and `Message` stay unchanged.

- [ ] **Step 4: Application: create, and name on the first turn**

In `backend/src/pharma_agent/application/conversation/queries.py` add after `get_conversation`:

```python
    async def create(self, user_id: str) -> ConversationView:
        conversation = Conversation.create_empty(
            user_id=user_id, now=self._clock.now()
        )
        await self._conversations.create(conversation)
        return ConversationView.of(conversation)
```

In `backend/src/pharma_agent/application/chat/service.py` replace the `else:` branch of `ChatService.open_turn` (lines 159-173) with:

```python
        else:
            found = await self._conversations.get(user_id, conversation_id)
            if found is None:
                raise ConversationNotFound(conversation_id)
            conversation = found
            # A conversation created by POST /conversations gets its title from
            # the first question, unless the user renamed it first.
            if conversation.title_from_first_message(message):
                await self._conversations.update_title(conversation)
            turns = await self._conversations.recent_turns(
                conversation_id, self._policy.context_turns
            )
            context = context_for_rephrase(
                conversation.summary,
                turns,
                max_turns=self._policy.context_turns,
                max_chars=self._policy.context_chars,
            )
            created = False
```

The title is written with `update_title` rather than inside `append_turn`, so a rename that happens while the turn streams is never overwritten by a stale title. If the turn then fails to persist, the conversation keeps the new title and still has `turn_count = 0`, so it stays out of the list.

- [ ] **Step 5: API route**

In `backend/src/pharma_agent/api/routers/conversations.py` add as the first route inside `build_conversations_router`:

```python
    @router.post("", response_model=ConversationView, status_code=201)
    async def create_conversation(
        user_id: UserId, container: ContainerDep
    ) -> ConversationView:
        return await container.queries.create(user_id)
```

- [ ] **Step 6: Run the tests to see them pass**

Run: `uv run pytest tests/domain/test_conversation.py tests/application/test_queries.py tests/application/test_chat_service.py tests/api/test_conversations_api.py -q`
Expected: PASS.

- [ ] **Step 7: Run the full check**

Run: `uv run ruff format src tests && uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q && uv run pytest -q -m integration`
Expected: all green; the architecture test still passes because the router only calls `ConversationQueries`.

- [ ] **Step 8: Commit**

```bash
git add backend/src/pharma_agent/domain/conversation/models.py backend/src/pharma_agent/application/conversation/queries.py backend/src/pharma_agent/application/chat/service.py backend/src/pharma_agent/api/routers/conversations.py backend/tests/domain/test_conversation.py backend/tests/application/test_queries.py backend/tests/application/test_chat_service.py backend/tests/api/test_conversations_api.py
git commit -m "feat(api): create conversations before the first message"
```

The commit message ends with the session attribution trailer.

---

### Task 5: OpenAPI operation ids, problem responses and `export-openapi`

**Files:**
- Modify: `backend/src/pharma_agent/api/problems.py` (append `PROBLEM_SCHEMA_REF`, `problem_responses`)
- Create: `backend/src/pharma_agent/api/openapi.py`
- Modify: `backend/src/pharma_agent/api/app.py` (imports lines 6-18, `FastAPI(...)` line 35)
- Modify: `backend/src/pharma_agent/api/routers/chat.py` (import block, `APIRouter` line 30)
- Modify: `backend/src/pharma_agent/api/routers/conversations.py` (import block, `APIRouter(...)` in `build_conversations_router`)
- Modify: `backend/src/pharma_agent/api/routers/feedback.py` (import block, `APIRouter` line 13)
- Modify: `backend/src/pharma_agent/api/routers/skills.py` (import block, `APIRouter` line 14)
- Modify: `backend/src/pharma_agent/api/routers/health.py` (import block, `APIRouter` line 11)
- Modify: `backend/src/pharma_agent/cli.py` (imports, new command after `migrate`)
- Create: `backend/tests/api/test_openapi.py`
- Modify: `backend/tests/test_cli.py` (imports, new test appended)

**Interfaces:**
- Consumes: `Problem`, `HealthResponse`, `PROBLEM_MEDIA_TYPE` (Task 1); `ConversationPage`, `MessagePage` (Task 3); `create_conversation` (Task 4); `AuthSettings(jwt_secret: SecretStr | None, google_client_id: str | None, google_client_secret: SecretStr | None)`; FastAPI 0.141 `FastAPI.openapi(self) -> dict[str, Any]`, which caches `openapi_schema` and regenerates it when routes change.
- Produces:
  - `pharma_agent.api.problems.PROBLEM_SCHEMA_REF = "#/components/schemas/Problem"`; `problem_responses(*statuses: int) -> dict[int | str, dict[str, Any]]`.
  - `pharma_agent.api.openapi`: `use_problem_details(document: dict[str, Any]) -> dict[str, Any]` (idempotent, in place); `class PharmaAgentAPI(FastAPI)` overriding `openapi`; `openapi_export_settings() -> Settings`; `render_openapi(app: FastAPI) -> str` (JSON, `indent=2`, `sort_keys=True`, `ensure_ascii=False`, trailing newline).
  - `create_app` builds `PharmaAgentAPI(..., generate_unique_id_function=lambda route: route.name)`.
  - CLI `pharma-agent export-openapi --output PATH` (creates parent directories, prints `wrote PATH`). P8 runs it as `uv run --directory ../backend pharma-agent export-openapi --output ../frontend/openapi.json`.
  - `tests/api/test_openapi.py::EXPECTED_OPERATION_IDS`, which P6 and P7 extend.

The post-processing is FastAPI's documented extension point for the OpenAPI document (overriding `openapi`). It does three things. (1) It moves every 4xx/5xx response to `application/problem+json` with `Problem`, which also covers the `ErrorModel` and `HTTPValidationError` responses declared by fastapi-users and FastAPI itself. (2) It renames form-body components from their titles, because `route.name` ids such as `auth:jwt.login` otherwise produce the component `login`, or `fastapi___compat__v2__Body_auth_jwt__login` once P7 adds a second auth router. (3) It drops the error schemas no response references any more.

- [ ] **Step 1: Write the failing tests**

`backend/tests/api/test_openapi.py`:

```python
import re
from collections import Counter
from typing import Any

from pharma_agent.api.app import create_app
from pharma_agent.api.openapi import openapi_export_settings

PROBLEM_CONTENT = {"schema": {"$ref": "#/components/schemas/Problem"}}

# Route function names (overview §3.5). P6 adds get_message_citation; P7 adds the
# cookie auth routes and moves Google OAuth to the cookie backend.
EXPECTED_OPERATION_IDS = {
    "health",
    "chat",
    "chat_stream",
    "create_conversation",
    "list_conversations",
    "get_conversation",
    "list_messages",
    "rename_conversation",
    "delete_conversation",
    "submit_feedback",
    "list_skills",
    "upload_skill",
    "set_skill_enabled",
    "delete_skill",
    "auth:jwt.login",
    "auth:jwt.logout",
    "register:register",
    "users:current_user",
    "users:patch_current_user",
    "users:user",
    "users:patch_user",
    "users:delete_user",
    "oauth:google.jwt.authorize",
    "oauth:google.jwt.callback",
}


def document() -> dict[str, Any]:
    return create_app(openapi_export_settings()).openapi()


def operations(doc: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    return [
        (method, path, operation)
        for path, item in doc["paths"].items()
        for method, operation in item.items()
    ]


def test_operation_ids_are_unique_route_names() -> None:
    ids = [operation["operationId"] for _, _, operation in operations(document())]
    assert [name for name, count in Counter(ids).items() if count > 1] == []
    assert set(ids) == EXPECTED_OPERATION_IDS


def test_every_error_response_is_a_problem() -> None:
    doc = document()
    for method, path, operation in operations(doc):
        for status, response in operation["responses"].items():
            if int(status) < 400:
                continue
            content = response["content"]
            assert content["application/problem+json"] == PROBLEM_CONTENT, (
                method,
                path,
                status,
            )
            if (path, status) != ("/api/v1/health", "503"):
                assert set(content) == {"application/problem+json"}, (
                    method,
                    path,
                    status,
                )

    health = doc["paths"]["/api/v1/health"]["get"]["responses"]["503"]["content"]
    assert health["application/json"] == {
        "schema": {"$ref": "#/components/schemas/HealthResponse"}
    }
    conversations = doc["paths"]["/api/v1/conversations"]
    assert {"401", "422", "503"} <= set(conversations["get"]["responses"])
    assert conversations["post"]["responses"]["201"]["content"]["application/json"] == {
        "schema": {"$ref": "#/components/schemas/ConversationView"}
    }
    assert conversations["get"]["responses"]["200"]["content"]["application/json"] == {
        "schema": {"$ref": "#/components/schemas/ConversationPage"}
    }
    skills = doc["paths"]["/api/v1/skills"]["post"]["responses"]
    assert {"401", "409", "413", "422", "503"} <= set(skills)
    register = doc["paths"]["/api/v1/auth/register"]["post"]["responses"]
    assert register["400"]["content"] == {"application/problem+json": PROBLEM_CONTENT}


def test_components_are_clean() -> None:
    schemas = document()["components"]["schemas"]
    assert {
        "Problem",
        "ProblemItem",
        "ConversationView",
        "ConversationPage",
        "MessagePage",
        "FeedbackView",
        "SkillView",
        "HealthResponse",
        "Body_auth_jwt_login",
        "Body_upload_skill",
    } <= set(schemas)
    assert not {"HTTPValidationError", "ValidationError", "ErrorModel"} & set(schemas)
    # OpenAPI 3.1 component keys must match ^[a-zA-Z0-9.\-_]+$.
    assert all(re.fullmatch(r"[A-Za-z0-9.\-_]+", name) for name in schemas)
    assert schemas["Problem"]["required"] == ["type", "title", "status", "code"]
    assert schemas["ProblemItem"]["required"] == ["loc", "message", "type"]


def test_post_processing_is_idempotent() -> None:
    app = create_app(openapi_export_settings())
    assert app.openapi() == app.openapi()
```

In `backend/tests/test_cli.py` add the imports `from pathlib import Path`, `from pharma_agent.api.app import create_app` and `from pharma_agent.api.openapi import openapi_export_settings, render_openapi`, then append:

```python
def test_export_openapi_is_deterministic_and_matches_the_app(tmp_path: Path) -> None:
    nested, flat = tmp_path / "frontend" / "openapi.json", tmp_path / "openapi.json"
    for output in (nested, flat):
        result = CliRunner().invoke(cli.app, ["export-openapi", "--output", str(output)])
        assert result.exit_code == 0, result.output
        assert f"wrote {output}" in result.output

    text = nested.read_text(encoding="utf-8")
    assert nested.read_bytes() == flat.read_bytes()
    assert text == render_openapi(create_app(openapi_export_settings()))
    assert text.endswith("\n")
    document = json.loads(text)
    assert list(document) == sorted(document)
    authorize = document["paths"]["/api/v1/auth/google/authorize"]["get"]
    assert authorize["operationId"] == "oauth:google.jwt.authorize"
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/api/test_openapi.py tests/test_cli.py -q`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'pharma_agent.api.openapi'`.

- [ ] **Step 3: Declare problem responses**

Append to `backend/src/pharma_agent/api/problems.py` (add `from http import HTTPStatus` and `from typing import Any` to its imports):

```python
PROBLEM_SCHEMA_REF = "#/components/schemas/Problem"


def problem_responses(*statuses: int) -> dict[int | str, dict[str, Any]]:
    """OpenAPI `responses` declaring `Problem` for each status.

    `PharmaAgentAPI` moves them from application/json to application/problem+json.
    """
    return {
        status: {"model": Problem, "description": HTTPStatus(status).phrase}
        for status in statuses
    }
```

Add `from pharma_agent.api.problems import problem_responses` to each router and declare the statuses on the router:

`backend/src/pharma_agent/api/routers/chat.py`:

```python
    router = APIRouter(
        prefix="/chat", tags=["chat"], responses=problem_responses(401, 404, 422, 503)
    )
```

`backend/src/pharma_agent/api/routers/conversations.py`:

```python
    router = APIRouter(
        prefix="/conversations",
        tags=["conversations"],
        responses=problem_responses(401, 404, 422, 503),
    )
```

`backend/src/pharma_agent/api/routers/feedback.py`:

```python
    router = APIRouter(
        prefix="/messages",
        tags=["feedback"],
        responses=problem_responses(401, 404, 422, 503),
    )
```

`backend/src/pharma_agent/api/routers/skills.py`:

```python
    router = APIRouter(
        prefix="/skills",
        tags=["skills"],
        responses=problem_responses(401, 404, 409, 413, 422, 503),
    )
```

`backend/src/pharma_agent/api/routers/health.py` (imports become `import asyncio`, `from typing import Any`, `from fastapi import APIRouter`, `from fastapi.responses import JSONResponse`, `from pharma_agent.api.deps import ContainerDep`, `from pharma_agent.api.problems import PROBLEM_MEDIA_TYPE, PROBLEM_SCHEMA_REF`, `from pharma_agent.api.schemas import HealthResponse`):

```python
# 503 is either a degraded HealthResponse or a SERVICE_STARTING problem.
HEALTH_RESPONSES: dict[int | str, dict[str, Any]] = {
    503: {
        "model": HealthResponse,
        "description": "A dependency is down, or the service is still starting",
        "content": {PROBLEM_MEDIA_TYPE: {"schema": {"$ref": PROBLEM_SCHEMA_REF}}},
    }
}


def build_health_router() -> APIRouter:
    router = APIRouter(tags=["health"], responses=HEALTH_RESPONSES)
```

The body of `health` stays unchanged.

- [ ] **Step 4: Post-process the OpenAPI document and export it**

`backend/src/pharma_agent/api/openapi.py`:

```python
"""OpenAPI document of the HTTP API, consistent with the RFC 9457 error handlers."""

import json
import re
from typing import Any, override

from fastapi import FastAPI
from pydantic import SecretStr

from pharma_agent.api.problems import PROBLEM_MEDIA_TYPE, PROBLEM_SCHEMA_REF
from pharma_agent.infrastructure.settings import AuthSettings, Settings

REF_PREFIX = "#/components/schemas/"
# Error schemas declared by FastAPI and fastapi-users that the handlers never send.
REPLACED_ERROR_SCHEMAS = ("HTTPValidationError", "ValidationError", "ErrorModel")
ERROR_SCHEMA_REFS = frozenset(
    {PROBLEM_SCHEMA_REF, REF_PREFIX + "HTTPValidationError", REF_PREFIX + "ErrorModel"}
)
UNSAFE_NAME_CHARACTERS = re.compile(r"[^A-Za-z0-9_]+")
EXPORT_PLACEHOLDER = "openapi-export"


def _use_problem_responses(document: dict[str, Any]) -> None:
    for path_item in document.get("paths", {}).values():
        for operation in path_item.values():
            for status, response in operation.get("responses", {}).items():
                if not (status.isdigit() and int(status) >= 400):
                    continue
                content = response.setdefault("content", {})
                json_body = content.get("application/json")
                if (
                    json_body is not None
                    and json_body.get("schema", {}).get("$ref") in ERROR_SCHEMA_REFS
                ):
                    del content["application/json"]
                content[PROBLEM_MEDIA_TYPE] = {"schema": {"$ref": PROBLEM_SCHEMA_REF}}


def _replace_refs(node: object, renames: dict[str, str]) -> None:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith(REF_PREFIX):
            name = ref.removeprefix(REF_PREFIX)
            if name in renames:
                node["$ref"] = REF_PREFIX + renames[name]
        for value in node.values():
            _replace_refs(value, renames)
    elif isinstance(node, list):
        for item in node:
            _replace_refs(item, renames)


def _name_body_schemas(document: dict[str, Any]) -> None:
    """Name form bodies `Body_<operation id>` with only [A-Za-z0-9_] characters.

    FastAPI titles them after the operation id; for ids like `auth:jwt.login` it
    shortens the component to `login`, or module-qualifies it on a name clash.
    """
    schemas: dict[str, Any] = document.get("components", {}).get("schemas", {})
    renames: dict[str, str] = {}
    for name, schema in schemas.items():
        title = schema.get("title")
        if isinstance(title, str) and title.startswith("Body_"):
            wanted = UNSAFE_NAME_CHARACTERS.sub("_", title)
            if wanted != name:
                renames[name] = wanted
    for old, new in renames.items():
        schemas[new] = {**schemas.pop(old), "title": new}
    _replace_refs(document, renames)


def _drop_replaced_error_schemas(document: dict[str, Any]) -> None:
    schemas: dict[str, Any] = document.get("components", {}).get("schemas", {})
    for name in REPLACED_ERROR_SCHEMAS:
        schema = schemas.pop(name, None)
        if schema is not None and f'"{REF_PREFIX}{name}"' in json.dumps(document):
            schemas[name] = schema


def use_problem_details(document: dict[str, Any]) -> dict[str, Any]:
    """Rewrite the generated document in place; safe to run on an already rewritten one."""
    _use_problem_responses(document)
    _name_body_schemas(document)
    _drop_replaced_error_schemas(document)
    return document


class PharmaAgentAPI(FastAPI):
    """FastAPI whose OpenAPI document describes the problem+json errors it sends."""

    @override
    def openapi(self) -> dict[str, Any]:
        # FastAPI caches the document and rebuilds it when routes change, so the
        # rewrite runs on every call and must be idempotent.
        return use_problem_details(super().openapi())


def openapi_export_settings() -> Settings:
    """Settings used only to render the document: nothing is served or connected.

    Google OAuth is enabled with placeholders so its routes are always documented.
    """
    return Settings(
        _env_file=None,
        auth=AuthSettings(
            jwt_secret=SecretStr(EXPORT_PLACEHOLDER + "-" + "0" * 32),
            google_client_id=EXPORT_PLACEHOLDER,
            google_client_secret=SecretStr(EXPORT_PLACEHOLDER),
        ),
    )


def render_openapi(app: FastAPI) -> str:
    return (
        json.dumps(app.openapi(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
```

In `backend/src/pharma_agent/api/app.py` keep `from fastapi import APIRouter, FastAPI` (`FastAPI` stays the return type of `create_app`), add `from pharma_agent.api.openapi import PharmaAgentAPI`, and replace line 35 with:

```python
    app = PharmaAgentAPI(
        title="Pharma Agent API",
        version="0.1.0",
        lifespan=lifespan,
        generate_unique_id_function=lambda route: route.name,
    )
```

In `backend/src/pharma_agent/cli.py` add `from pathlib import Path` to the imports and add after the `migrate` command:

```python
@app.command("export-openapi")
def export_openapi(
    output: Path = typer.Option(
        ...,
        "--output",
        dir_okay=False,
        help="File to write, for example ../frontend/openapi.json",
    ),
) -> None:
    """Ghi tài liệu OpenAPI của HTTP API ra file, không cần chạy server."""
    from pharma_agent.api.app import create_app
    from pharma_agent.api.openapi import openapi_export_settings, render_openapi

    document = render_openapi(create_app(openapi_export_settings()))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8", newline="\n")
    typer.echo(f"wrote {output}")
```

`create_app` does not enter the lifespan, so no database, Qdrant or LLM is touched.

- [ ] **Step 5: Run the tests to see them pass**

Run: `uv run pytest tests/api/test_openapi.py tests/test_cli.py tests/api -q`
Expected: PASS. FastAPI warns `Duplicate Operation ID ...` whenever two routes share a name, and `filterwarnings = ["error"]` turns that warning into a failure as well.

- [ ] **Step 6: Run the full check**

Run: `uv run ruff format src tests && uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check --min-severity warn && uv run pytest -q && uv run pytest -q -m integration`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add backend/src/pharma_agent/api/problems.py backend/src/pharma_agent/api/openapi.py backend/src/pharma_agent/api/app.py backend/src/pharma_agent/api/routers/chat.py backend/src/pharma_agent/api/routers/conversations.py backend/src/pharma_agent/api/routers/feedback.py backend/src/pharma_agent/api/routers/skills.py backend/src/pharma_agent/api/routers/health.py backend/src/pharma_agent/cli.py backend/tests/api/test_openapi.py backend/tests/test_cli.py
git commit -m "feat(api): pin operation ids, document problem responses, export OpenAPI"
```

The commit message ends with the session attribution trailer.

---

## Self-Review

### Spec coverage

| Spec requirement | Task |
| --- | --- |
| §4.1 `POST /conversations` creates "Cuộc trò chuyện mới", 201 `ConversationView` | Task 4 |
| §4.1 first turn sets the title from the first message while the title is still the default | Task 4 |
| §4.1 `GET /conversations?limit=20&cursor=` → `{items, next_cursor}`, `(updated_at desc, id desc)`, excludes `turn_count = 0` | Task 3 |
| §4.1 `GET /conversations/{id}`, `PATCH`, `DELETE` unchanged | Task 3 (router kept), Task 5 (documented problem responses) |
| §4.1 `GET /conversations/{id}/messages?limit=30&cursor=`: newest page first, items oldest → newest, `next_cursor` to older | Task 3 (`items` are `MessageView`; P6 switches them to `UIMessage`) |
| §4.1 cursor = base64url JSON `{"t","id"}`, keyset `(updated_at, id)` / `(created_at, id)`, malformed → 422 `INVALID_CURSOR` | Task 2 (helpers), Task 3 (queries, routes, API test) |
| §4.1 `limit` at most 100 for both lists | Task 3 |
| §7 every error is `application/problem+json` with `type`, `title`, `status`, `detail`, `code` | Task 1 |
| §7 `errors` only on 422 | Task 1 (`VALIDATION_ERROR`) |
| §7 application errors keep their codes (`CONVERSATION_NOT_FOUND`, `MESSAGE_NOT_FOUND`, `SKILL_NOT_FOUND`, `SKILL_NAME_TAKEN`, `PAYLOAD_TOO_LARGE`, `INVALID_INPUT`, `INVALID_CURSOR`, `AGENT_UNAVAILABLE`) | Task 1, Task 2 |
| §7 fastapi-users `HTTPException` codes (`REGISTER_USER_ALREADY_EXISTS`, `REGISTER_INVALID_PASSWORD`, `LOGIN_BAD_CREDENTIALS`, `OAUTH_*`) become `code` | Task 1 (unit through ASGI plus real fastapi-users in `test_e2e_postgres.py`) |
| §7 `RequestValidationError` → 422 `VALIDATION_ERROR` with `errors[{loc,message,type}]` | Task 1 |
| §7 401 → `UNAUTHORIZED` | Task 1 |
| §7 CSRF 403 → `CSRF_FAILED` | P7 (starlette-csrf is added there) |
| §7 error responses declared in OpenAPI | Task 5 |
| §8 `generate_unique_id_function=lambda route: route.name` and a uniqueness test | Task 5 |
| §8 `pharma-agent export-openapi --output <path>` without a server, test that the export matches the app | Task 5 |
| §8 `/chat/stream` declares `text/event-stream`; `UIMessage`, parts, `MessageMetadata`, `EvidenceItem`, `CitationDetail` in components | P6 |
| §8 `Problem` in components | Task 5 |
| §10 "Phân trang": ties on `created_at` give no duplicates and no gaps; bad cursor → 422 | Task 3 (in-memory, Postgres and HTTP) |
| §10 "Lỗi": problem+json per error group | Task 1 |
| §10 "OpenAPI": unique operation ids, export matches | Task 5 |
| Extra: 503 from dependencies get stable codes (`SERVICE_STARTING`, `SERVICE_NOT_CONFIGURED`), Starlette 404/405, unhandled 500 `INTERNAL_ERROR` | Task 1 |
| Extra: migration `0007` keyset indexes | Task 3 |

### Names checked against the overview

- `pharma_agent.application.pagination.encode_cursor(timestamp: datetime, id: str) -> str`, `decode_cursor(cursor: str) -> tuple[datetime, str]`, `InvalidCursor` with code `INVALID_CURSOR`: Task 2.
- Response models `ConversationView`, `ConversationPage`, `MessagePage`, `FeedbackView`, `SkillView`, `HealthResponse`, `Problem`, `ProblemItem`: Tasks 1, 3 and 5; `test_components_are_clean` asserts them.
- Error `type` `urn:pharma-agent:problem:<lower kebab code>`: `problem_type` in Task 1.
- Operation ids: route function names `health`, `chat`, `chat_stream`, `create_conversation`, `list_conversations`, `get_conversation`, `list_messages`, `rename_conversation`, `delete_conversation`, `submit_feedback`, `list_skills`, `upload_skill`, `set_skill_enabled`, `delete_skill` and the fastapi-users names: Task 5.
- CLI `pharma-agent export-openapi --output PATH`: Task 5.
- Alembic revision `0007` with down revision `0006`: Task 3.

### Deviations and decisions

- The overview's pinned list includes `get_message_citation`, `auth:cookie.login`, `auth:cookie.logout` and `oauth:google.cookie.*`. Those routes only exist after P6/P7, so `EXPECTED_OPERATION_IDS` here has the current `oauth:google.jwt.*` ids, and P6/P7 must update that set.
- With `route.name` ids, FastAPI names the fastapi-users login form component `login`, or `fastapi___compat__v2__Body_auth_jwt__login` once P7 adds the cookie router. `PharmaAgentAPI` renames it to `Body_auth_<backend>_login`; this was verified against FastAPI 0.141.1 with both one and two auth routers.
- FastAPI always emits `responses={..: {"model": Problem}}` under `application/json`, and fastapi-users documents its 400s with `ErrorModel`. The document is therefore post-processed so every 4xx/5xx says `application/problem+json`.
- The degraded `/health` 503 stays a `HealthResponse` status report (documented as both `application/json` and problem) because monitoring reads its `checks`.
- Conversation and message ids are 32-hex strings, not hyphenated UUIDs: cursors carry the hex id, and `decode_cursor` accepts both forms.
- The repository port change forces the port, repositories, queries and routes into one task (Task 3), so every commit stays green.
- The first-message title is persisted with `update_title` in `open_turn`, not inside `append_turn`, so a rename during a streaming turn is never overwritten.
- Line numbers refer to commit `8f76523`; P1-P3 may shift them, so each step also names the function it edits.

