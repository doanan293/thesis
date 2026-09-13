"""fastapi-users wiring: cookie sessions for browsers, JWT bearer for CLI and tests,
registration, current user and Google OAuth."""

import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from fastapi_users import BaseUserManager, FastAPIUsers, UUIDIDMixin, schemas
from fastapi_users.authentication import (
    AuthenticationBackend,
    BearerTransport,
    CookieTransport,
    JWTStrategy,
)
from fastapi_users.authentication.strategy.db import DatabaseStrategy
from fastapi_users_db_sqlalchemy import SQLAlchemyUserDatabase
from fastapi_users_db_sqlalchemy.access_token import SQLAlchemyAccessTokenDatabase
from httpx_oauth.clients.google import GoogleOAuth2
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pharma_agent.infrastructure.persistence.postgres.tables import (
    AccessTokenTable,
    OAuthAccountTable,
    UserTable,
)
from pharma_agent.infrastructure.settings import AuthSettings

LOGIN_URL = "/api/v1/auth/jwt/login"
SESSION_COOKIE_NAME = "pharma_session"

SessionFactoryResolver = Callable[[Request], async_sessionmaker[AsyncSession]]
UserDatabase = SQLAlchemyUserDatabase[UserTable, uuid.UUID]
AccessTokenDatabase = SQLAlchemyAccessTokenDatabase[AccessTokenTable]


class UserRead(schemas.BaseUser[uuid.UUID]):
    display_name: str = ""


class UserCreate(schemas.BaseUserCreate):
    display_name: str = ""


class UserUpdate(schemas.BaseUserUpdate):
    display_name: str | None = None


class UserManager(UUIDIDMixin, BaseUserManager[UserTable, uuid.UUID]):
    def __init__(self, user_db: UserDatabase, secret: str) -> None:
        super().__init__(user_db)
        self.reset_password_token_secret = secret
        self.verification_token_secret = secret


@dataclass(frozen=True)
class Auth:
    users: FastAPIUsers[UserTable, uuid.UUID]
    cookie_backend: AuthenticationBackend[UserTable, uuid.UUID]
    jwt_backend: AuthenticationBackend[UserTable, uuid.UUID]
    current_active_user: Callable[..., Any]
    google: GoogleOAuth2 | None
    secret: str
    frontend_url: str
    cookie_secure: bool


def build_auth(
    settings: AuthSettings, resolve_sessions: SessionFactoryResolver
) -> Auth:
    secret = settings.require_jwt_secret()

    async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
        async with resolve_sessions(request)() as session:
            yield session

    async def get_user_db(
        session: Annotated[AsyncSession, Depends(get_session)],
    ) -> AsyncIterator[UserDatabase]:
        yield SQLAlchemyUserDatabase(session, UserTable, OAuthAccountTable)

    async def get_access_token_db(
        session: Annotated[AsyncSession, Depends(get_session)],
    ) -> AsyncIterator[AccessTokenDatabase]:
        yield SQLAlchemyAccessTokenDatabase(session, AccessTokenTable)

    async def get_user_manager(
        user_db: Annotated[UserDatabase, Depends(get_user_db)],
    ) -> AsyncIterator[UserManager]:
        yield UserManager(user_db, secret)

    def get_session_strategy(
        access_tokens: Annotated[AccessTokenDatabase, Depends(get_access_token_db)],
    ) -> DatabaseStrategy[UserTable, uuid.UUID, AccessTokenTable]:
        return DatabaseStrategy[UserTable, uuid.UUID, AccessTokenTable](
            access_tokens, lifetime_seconds=settings.session_lifetime_seconds
        )

    def get_jwt_strategy() -> JWTStrategy[UserTable, uuid.UUID]:
        return JWTStrategy(
            secret=secret, lifetime_seconds=settings.jwt_lifetime_seconds
        )

    cookie_backend = AuthenticationBackend(
        name="cookie",
        transport=CookieTransport(
            cookie_name=SESSION_COOKIE_NAME,
            cookie_max_age=settings.session_lifetime_seconds,
            cookie_secure=settings.cookie_secure,
            cookie_httponly=True,
            cookie_samesite="lax",
        ),
        get_strategy=get_session_strategy,
    )
    jwt_backend = AuthenticationBackend(
        name="jwt",
        transport=BearerTransport(tokenUrl=LOGIN_URL),
        get_strategy=get_jwt_strategy,
    )
    # False positive: pyrefly 1.3.0 orders the type parameters of the generic alias
    # `UserManagerDependency[UP, ID]` by TypeVar declaration order (ID, UP) instead of
    # first appearance, so it expects BaseUserManager[UUID, UserTable]. Types are correct.
    users = FastAPIUsers[UserTable, uuid.UUID](
        get_user_manager,  # pyrefly: ignore[bad-argument-type]
        [cookie_backend, jwt_backend],
    )
    google = None
    if (
        settings.google_enabled
        and settings.google_client_id
        and settings.google_client_secret
    ):
        google = GoogleOAuth2(
            settings.google_client_id, settings.google_client_secret.get_secret_value()
        )
    return Auth(
        users=users,
        cookie_backend=cookie_backend,
        jwt_backend=jwt_backend,
        current_active_user=users.current_user(active=True),
        google=google,
        secret=secret,
        frontend_url=settings.frontend_url,
        cookie_secure=settings.cookie_secure,
    )


def include_auth_routes(router: APIRouter, auth: Auth) -> None:
    router.include_router(
        auth.users.get_auth_router(auth.cookie_backend),
        prefix="/auth/cookie",
        tags=["auth"],
    )
    router.include_router(
        auth.users.get_auth_router(auth.jwt_backend), prefix="/auth/jwt", tags=["auth"]
    )
    router.include_router(
        auth.users.get_register_router(UserRead, UserCreate),
        prefix="/auth",
        tags=["auth"],
    )
    router.include_router(
        auth.users.get_users_router(UserRead, UserUpdate),
        prefix="/users",
        tags=["users"],
    )
    if auth.google is not None:
        router.include_router(
            auth.users.get_oauth_router(
                auth.google,
                auth.cookie_backend,
                auth.secret,
                redirect_url=f"{auth.frontend_url.rstrip('/')}/auth/google/callback",
                associate_by_email=True,
                is_verified_by_default=True,
                csrf_token_cookie_secure=auth.cookie_secure,
            ),
            prefix="/auth/google",
            tags=["auth"],
        )
