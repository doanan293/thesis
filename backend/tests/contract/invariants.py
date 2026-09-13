"""Stream guarantees of spec A §3.5 and the ordering rules of §3.2, on decoded chunks."""

import re
from typing import Any

from tests.contract.ui_chunks import (
    DATA_PAYLOADS,
    FinishMetadata,
    PharmaSource,
    validate_chunk,
)

MARKER = re.compile(r"\[(\d{1,3})\]")
PARTIAL_MARKER_AT_END = re.compile(r"\[\d{0,3}$")


def assert_stream_invariants(chunks: list[dict[str, Any]]) -> None:
    assert chunks, "empty stream"
    for chunk in chunks:
        validate_chunk(chunk)
        payload = DATA_PAYLOADS.get(chunk["type"])
        if payload is not None:
            payload.model_validate(chunk["data"])

    types = [chunk["type"] for chunk in chunks]
    assert types[:2] == ["start", "data-conversation"], types
    assert types[-1] == "finish" and types.count("finish") == 1, types
    assert "error" not in types, types
    FinishMetadata.model_validate(chunks[-1]["messageMetadata"])

    deltas = [chunk["delta"] for chunk in chunks if chunk["type"] == "text-delta"]
    for delta in deltas:
        assert PARTIAL_MARKER_AT_END.search(delta) is None, f"marker split: {delta!r}"
    if deltas:
        assert types.count("text-start") == 1 and types.count("text-end") == 1, types
        last_delta = max(i for i, kind in enumerate(types) if kind == "text-delta")
        assert types.index("text-start") < types.index(
            "text-delta"
        ) and last_delta < types.index("text-end"), types
    else:
        assert "text-start" not in types and "text-end" not in types, types

    sources = [chunk for chunk in chunks if chunk["type"] == "source-document"]
    if sources:
        assert "text-end" in types, types
        assert types.index("text-end") < types.index("source-document"), types
    indexes: list[int] = []
    for source in sources:
        pharma = PharmaSource.model_validate(source["providerMetadata"]["pharma"])
        assert pharma.isCurrent is True
        assert source["mediaType"] == "text/markdown" and source["sourceId"]
        indexes.append(pharma.index)
    cited = {int(number) for number in MARKER.findall("".join(deltas))}
    assert len(indexes) == len(set(indexes)), f"duplicate source index: {indexes}"
    assert set(indexes) == cited, (
        f"markers {sorted(cited)} vs sources {sorted(indexes)}"
    )
