"""Cookie sessions (DatabaseStrategy over access_tokens) and bearer JWT against real Postgres."""

import pytest
from sqlalchemy import text

from pharma_agent.infrastructure.persistence.postgres.database import Database
from tests.api.postgres_app import (
    PASSWORD,
    cookie_login,
    csrf_header,
    new_email,
    postgres_app,
    register,
    set_cookie,
)

pytestmark = pytest.mark.integration


async def sessions_for(database: Database, email: str) -> int:
    async with database.sessions() as session:
        result = await session.execute(
            text(
                'SELECT count(*) FROM access_tokens AS t JOIN "user" AS u '
                "ON u.id = t.user_id WHERE u.email = :email"
            ),
            {"email": email},
        )
        return int(result.scalar_one())


async def test_cookie_login_sets_an_httponly_session_that_authorizes(
    migrated_dsn: str,
) -> None:
    email = new_email()
    async with postgres_app(migrated_dsn) as (client, database):
        await register(client, email)
        login = await cookie_login(client, email)
        session_cookie = set_cookie(login, "pharma_session")
        assert "httponly" in session_cookie and "secure" in session_cookie
        assert "samesite=lax" in session_cookie and "max-age=604800" in session_cookie
        assert await sessions_for(database, email) == 1
        me = await client.get("/api/v1/users/me")
        assert me.status_code == 200 and me.json()["email"] == email


async def test_cookie_logout_revokes_the_session_row(migrated_dsn: str) -> None:
    email = new_email()
    async with postgres_app(migrated_dsn) as (client, database):
        await register(client, email)
        await cookie_login(client, email)
        token = client.cookies["pharma_session"]
        logout = await client.post(
            "/api/v1/auth/cookie/logout", headers=csrf_header(client)
        )
        assert logout.status_code == 204, logout.text
        assert "max-age=0" in set_cookie(logout, "pharma_session")
        assert "pharma_session" not in client.cookies
        assert await sessions_for(database, email) == 0
        replay = await client.get(
            "/api/v1/users/me", headers={"Cookie": f"pharma_session={token}"}
        )
        assert replay.status_code == 401
        assert replay.json()["code"] == "UNAUTHORIZED"


async def test_cookie_session_needs_the_csrf_header_on_unsafe_requests(
    migrated_dsn: str,
) -> None:
    email = new_email()
    async with postgres_app(migrated_dsn) as (client, _):
        await register(client, email)
        await cookie_login(client, email)
        rejected = await client.patch("/api/v1/users/me", json={"display_name": "An"})
        assert rejected.status_code == 403
        assert rejected.json()["code"] == "CSRF_FAILED"
        accepted = await client.patch(
            "/api/v1/users/me",
            json={"display_name": "An"},
            headers=csrf_header(client),
        )
        assert accepted.status_code == 200, accepted.text
        assert accepted.json()["display_name"] == "An"


async def test_bearer_jwt_still_authorizes_without_csrf_or_sessions(
    migrated_dsn: str,
) -> None:
    email = new_email()
    async with postgres_app(migrated_dsn) as (client, database):
        await register(client, email)
        login = await client.post(
            "/api/v1/auth/jwt/login", data={"username": email, "password": PASSWORD}
        )
        assert login.status_code == 200, login.text
        headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
        client.cookies.clear()
        patched = await client.patch(
            "/api/v1/users/me", json={"display_name": "Bình"}, headers=headers
        )
        assert patched.status_code == 200, patched.text
        assert patched.json()["display_name"] == "Bình"
        assert await sessions_for(database, email) == 0
