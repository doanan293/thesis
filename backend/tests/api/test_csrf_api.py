"""CSRF double-submit protection (starlette-csrf) on the real app over in-memory services."""

import pytest
from starlette.middleware.cors import CORSMiddleware

from pharma_agent.api.app import create_app
from pharma_agent.api.csrf import ProblemCSRFMiddleware
from pharma_agent.infrastructure.settings import Settings
from tests.api.harness import build_harness, settings

CSRF_PROBLEM = {
    "type": "urn:pharma-agent:problem:csrf-failed",
    "title": "Csrf failed",
    "status": 403,
    "detail": "Send the value of the csrftoken cookie in the x-csrftoken header.",
    "code": "CSRF_FAILED",
}


async def test_unsafe_request_without_session_cookie_is_not_checked() -> None:
    harness = build_harness()
    async with harness.client() as client:
        response = await client.post("/api/v1/conversations")
    assert response.status_code == 201, response.text


async def test_session_cookie_without_csrf_header_is_a_problem() -> None:
    harness = build_harness()
    async with harness.client() as client:
        response = await client.post(
            "/api/v1/conversations", headers={"Cookie": "pharma_session=opaque"}
        )
    assert response.status_code == 403
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json() == CSRF_PROBLEM


async def test_session_cookie_with_matching_csrf_header_passes() -> None:
    harness = build_harness()
    async with harness.client() as client:
        safe = await client.get("/api/v1/health")
        token = safe.cookies["csrftoken"]
        response = await client.post(
            "/api/v1/conversations",
            headers={
                "Cookie": f"pharma_session=opaque; csrftoken={token}",
                "x-csrftoken": token,
            },
        )
    assert response.status_code == 201, response.text


async def test_forged_csrf_header_is_rejected() -> None:
    harness = build_harness()
    async with harness.client() as client:
        safe = await client.get("/api/v1/health")
        token = safe.cookies["csrftoken"]
        response = await client.post(
            "/api/v1/conversations",
            headers={
                "Cookie": f"pharma_session=opaque; csrftoken={token}",
                "x-csrftoken": "forged",
            },
        )
    assert response.status_code == 403
    assert response.json()["code"] == "CSRF_FAILED"


async def test_csrf_cookie_is_script_readable_and_secure() -> None:
    harness = build_harness()
    async with harness.client() as client:
        response = await client.get("/api/v1/health")
    set_cookie = response.headers["set-cookie"].lower()
    assert set_cookie.startswith("csrftoken=")
    assert "httponly" not in set_cookie
    assert "samesite=lax" in set_cookie
    assert "secure" in set_cookie and "path=/" in set_cookie


def _middleware(origins: list[str]) -> list[object]:
    base = settings()
    configured = base.model_copy(
        update={"api": base.api.model_copy(update={"cors_origins": origins})}
    )
    return [middleware.cls for middleware in create_app(configured).user_middleware]


def test_cors_is_off_by_default_and_outermost_when_configured() -> None:
    default = _middleware([])
    assert ProblemCSRFMiddleware in default and CORSMiddleware not in default
    configured = _middleware(["http://localhost:5173"])
    assert configured.index(CORSMiddleware) < configured.index(ProblemCSRFMiddleware)


def test_missing_csrf_secret_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PHARMA_AUTH__CSRF_SECRET", raising=False)
    with pytest.raises(ValueError, match="PHARMA_AUTH__CSRF_SECRET"):
        create_app(Settings(_env_file=None, auth={"jwt_secret": "s" * 40}))
