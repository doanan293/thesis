from typing import NoReturn

import pytest
from fastapi import APIRouter, HTTPException, Request
from fastapi_users.authentication import CookieTransport
from pydantic import SecretStr
from starlette.routing import NoMatchFound

from pharma_agent.infrastructure.auth.users import build_auth, include_auth_routes
from pharma_agent.infrastructure.settings import AuthSettings

SECRET = "s" * 40


def no_sessions(request: Request) -> NoReturn:
    """Session resolver for wiring tests that never touch the database."""
    raise HTTPException(status_code=503)


def auth_router(settings: AuthSettings) -> APIRouter:
    router = APIRouter()
    include_auth_routes(router, build_auth(settings, no_sessions))
    return router


# FastAPI keeps included routers as wrappers, so route names are resolved through the
# public `url_path_for` instead of iterating `router.routes`.
def test_routes_without_google() -> None:
    settings = AuthSettings(jwt_secret=SecretStr(SECRET))
    router = auth_router(settings)
    assert router.url_path_for("auth:cookie.login") == "/auth/cookie/login"
    assert router.url_path_for("auth:cookie.logout") == "/auth/cookie/logout"
    assert router.url_path_for("auth:jwt.login") == "/auth/jwt/login"
    assert router.url_path_for("auth:jwt.logout") == "/auth/jwt/logout"
    assert router.url_path_for("register:register") == "/auth/register"
    assert router.url_path_for("users:current_user") == "/users/me"
    for name in ("oauth:google.cookie.authorize", "oauth:google.cookie.callback"):
        with pytest.raises(NoMatchFound):
            router.url_path_for(name)
    auth = build_auth(settings, no_sessions)
    assert auth.google is None
    assert [backend.name for backend in auth.users.authenticator.backends] == [
        "cookie",
        "jwt",
    ]


def test_cookie_backend_follows_session_settings() -> None:
    auth = build_auth(
        AuthSettings(
            jwt_secret=SecretStr(SECRET),
            session_lifetime_seconds=3600,
            cookie_secure=False,
        ),
        no_sessions,
    )
    transport = auth.cookie_backend.transport
    assert isinstance(transport, CookieTransport)
    assert (
        transport.cookie_name,
        transport.cookie_max_age,
        transport.cookie_secure,
        transport.cookie_httponly,
        transport.cookie_samesite,
    ) == ("pharma_session", 3600, False, True, "lax")


def test_routes_with_google_use_the_cookie_backend() -> None:
    router = auth_router(
        AuthSettings(
            jwt_secret=SecretStr(SECRET),
            google_client_id="id",
            google_client_secret=SecretStr("secret"),
            frontend_url="https://app.example",
        )
    )
    assert (
        router.url_path_for("oauth:google.cookie.authorize") == "/auth/google/authorize"
    )
    assert (
        router.url_path_for("oauth:google.cookie.callback") == "/auth/google/callback"
    )


def test_missing_secret_fails_fast() -> None:
    with pytest.raises(ValueError, match="PHARMA_AUTH__JWT_SECRET"):
        build_auth(AuthSettings(), no_sessions)
