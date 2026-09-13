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
