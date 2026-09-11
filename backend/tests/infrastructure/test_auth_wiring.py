import pytest
from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import SecretStr

from pharma_agent.infrastructure.auth.users import build_auth, include_auth_routes
from pharma_agent.infrastructure.settings import AuthSettings

SECRET = "s" * 40


def no_sessions(request: Request):
    raise HTTPException(status_code=503)


def paths(router: APIRouter) -> set[str]:
    app = FastAPI()
    app.include_router(router)
    return set(app.openapi()["paths"])


def test_routes_without_google() -> None:
    auth = build_auth(AuthSettings(jwt_secret=SecretStr(SECRET)), no_sessions)
    router = APIRouter()
    include_auth_routes(router, auth)
    assert {
        "/auth/jwt/login",
        "/auth/jwt/logout",
        "/auth/register",
        "/users/me",
    } <= paths(router)
    assert not any(path.startswith("/auth/google") for path in paths(router))
    assert auth.google is None


def test_routes_with_google() -> None:
    auth = build_auth(
        AuthSettings(
            jwt_secret=SecretStr(SECRET),
            google_client_id="id",
            google_client_secret=SecretStr("secret"),
            frontend_url="https://app.example",
        ),
        no_sessions,
    )
    router = APIRouter()
    include_auth_routes(router, auth)
    assert {"/auth/google/authorize", "/auth/google/callback"} <= paths(router)


def test_missing_secret_fails_fast() -> None:
    with pytest.raises(ValueError, match="PHARMA_AUTH__JWT_SECRET"):
        build_auth(AuthSettings(), no_sessions)
