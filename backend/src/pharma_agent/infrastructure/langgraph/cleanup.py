"""Retention for LangGraph Postgres checkpoints (one thread per chat turn)."""

from psycopg import AsyncConnection

_DELETE_EXPIRED = """
WITH expired AS (
    DELETE FROM checkpoints
    WHERE (checkpoint->>'ts')::timestamptz < now() - make_interval(days => %s)
    RETURNING thread_id, checkpoint_ns, checkpoint_id
),
removed_writes AS (
    DELETE FROM checkpoint_writes AS w
    USING expired AS e
    WHERE w.thread_id = e.thread_id
      AND w.checkpoint_ns = e.checkpoint_ns
      AND w.checkpoint_id = e.checkpoint_id
)
SELECT count(*) FROM expired
"""

# Runs as a second statement: CTEs share one snapshot, so blobs can only be judged
# orphaned after the checkpoint rows are actually gone.
_DELETE_ORPHAN_BLOBS = """
DELETE FROM checkpoint_blobs AS b
WHERE NOT EXISTS (
    SELECT 1 FROM checkpoints AS c
    WHERE c.thread_id = b.thread_id AND c.checkpoint_ns = b.checkpoint_ns
)
"""


async def delete_expired_checkpoints(conninfo: str, *, retention_days: int) -> int:
    """Delete checkpoints older than `retention_days` with their writes and blobs.

    Returns the number of checkpoint rows removed.
    """
    async with (
        await AsyncConnection.connect(conninfo) as connection,
        connection.transaction(),
    ):
        cursor = await connection.execute(_DELETE_EXPIRED, (retention_days,))
        row = await cursor.fetchone()
        await connection.execute(_DELETE_ORPHAN_BLOBS)
    return int(row[0]) if row else 0
