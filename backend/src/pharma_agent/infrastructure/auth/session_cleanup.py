"""Retention for cookie sessions: fastapi-users ignores expired access tokens but never deletes them."""

from psycopg import AsyncConnection

_DELETE_EXPIRED = """
DELETE FROM access_tokens
WHERE created_at < now() - make_interval(secs => %s)
"""


async def delete_expired_access_tokens(conninfo: str, *, lifetime_seconds: int) -> int:
    """Delete access tokens older than `lifetime_seconds` and return how many were removed.

    `DatabaseStrategy` treats exactly these rows as expired (`created_at` before now minus
    the lifetime), so deleting them never logs out a valid session.
    """
    async with (
        await AsyncConnection.connect(conninfo) as connection,
        connection.transaction(),
    ):
        cursor = await connection.execute(_DELETE_EXPIRED, (lifetime_seconds,))
    return cursor.rowcount
