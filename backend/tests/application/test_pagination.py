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
    # Base64 is only padded when the byte count is not a multiple of 3; this layout
    # is 86 bytes, while json.dumps defaults (87) and compact separators (84) are not.
    payload = {"t": AT.isoformat(), "id": "12345678-1234-5678-1234-567812345678"}
    raw = json.dumps(payload, separators=(",", ": ")).encode()
    cursor = base64.urlsafe_b64encode(raw).decode()
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
