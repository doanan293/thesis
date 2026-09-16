import math

import pytest

from pharma_lab.runtime.benchmarking import (
    BenchmarkMeasurement,
    LevelResult,
    RerankGroup,
    check_score_consistency,
    describe_levels,
    percentile,
    recommend,
    rerank_groups_from_rows,
    sample_rerank_groups,
    stratified_sample,
)
from pharma_lab.runtime.runtime_profiles import reranker_candidate

LEVELS = tuple(
    reranker_candidate(
        server_slots=64, ubatch=ubatch, request_batch_size=30, concurrency=4
    )
    for ubatch in (8192, 16384, 32768)
)


def _measurement(
    level: int, *, seconds: float = 10.0, p95: float = 1.0
) -> BenchmarkMeasurement:
    return BenchmarkMeasurement(
        LEVELS[level],
        960,
        100_000,
        seconds,
        latency_p50_seconds=p95 / 2,
        latency_p95_seconds=p95,
    )


def _group(query_id: str, document_length: int) -> RerankGroup:
    return RerankGroup(
        query_id,
        "q",
        (f"{query_id}-0", f"{query_id}-1"),
        ("x" * document_length, "x" * document_length),
    )


def test_stratified_sample_is_stable_and_spans_lengths():
    items = [{"id": i, "length": i} for i in range(1, 101)]
    first = stratified_sample(
        items, 20, key=lambda x: x["id"], length=lambda x: x["length"]
    )
    second = stratified_sample(
        list(reversed(items)), 20, key=lambda x: x["id"], length=lambda x: x["length"]
    )
    assert [x["id"] for x in first] == [x["id"] for x in second]
    assert min(x["length"] for x in first) <= 5
    assert max(x["length"] for x in first) >= 95


def test_sample_rerank_groups_takes_one_group_per_character_stratum():
    groups = [_group(f"q{index:03d}", index) for index in range(1, 101)]

    first = sample_rerank_groups(groups, 6)

    assert first == sample_rerank_groups(list(reversed(groups)), 6)
    assert [len(group.documents[0]) for group in first] == [9, 26, 42, 59, 76, 92]


def test_sample_rerank_groups_returns_every_group_when_there_are_few():
    groups = [_group("b", 5), _group("a", 1)]

    assert sample_rerank_groups(groups, 6) == (groups[1], groups[0])


def test_rerank_groups_keep_only_full_query_groups():
    rows = [
        {
            "query_id": "q1",
            "query": "một",
            "candidates": [
                {"chunk_id": f"c{index}", "document_text": f"d{index}"}
                for index in range(3)
            ],
        },
        {
            "query_id": "q2",
            "query": "hai",
            "candidates": [{"chunk_id": "c0", "document_text": "d0"}],
        },
    ]

    groups = rerank_groups_from_rows(rows, documents_per_group=2)

    assert groups == [RerankGroup("q1", "một", ("c0", "c1"), ("d0", "d1"))]
    assert groups[0].score_keys() == ("q1\x1fc0", "q1\x1fc1")
    assert groups[0].characters == 2 * len("một") + 4


def test_throughput_objective_picks_the_most_pairs_per_second():
    measurements = [
        _measurement(0, seconds=12.0),
        _measurement(1, seconds=8.0),
        _measurement(2, seconds=9.0),
    ]

    assert recommend(measurements, objective="throughput") == LEVELS[1]


def test_latency_objective_picks_the_lowest_p95_then_throughput():
    measurements = [
        _measurement(0, seconds=9.0, p95=0.8),
        _measurement(1, seconds=8.0, p95=0.8),
        _measurement(2, seconds=5.0, p95=0.9),
    ]

    assert recommend(measurements, objective="latency") == LEVELS[1]


def test_recommend_skips_invalid_levels_and_needs_one_valid_level():
    invalid = BenchmarkMeasurement.invalid(
        LEVELS[1], "ModelServerExited", "CUDA out of memory"
    )

    assert (
        recommend([_measurement(0, seconds=12.0), invalid], objective="throughput")
        == LEVELS[0]
    )
    assert recommend([invalid], objective="throughput") is None
    assert recommend([invalid], objective="latency") is None


def test_score_check_keeps_levels_that_rank_the_candidates_the_same():
    """Scores shift with the batch shape on a GPU; only the ranking has to hold."""
    results = [
        LevelResult(BenchmarkMeasurement.invalid(LEVELS[0], "RuntimeError"), {}),
        LevelResult(_measurement(1), {"q\x1fa": 0.9, "q\x1fb": 0.1}),
        LevelResult(_measurement(2), {"q\x1fa": 0.9005, "q\x1fb": 0.1}),
        LevelResult(_measurement(0), {"q\x1fa": 0.82, "q\x1fb": 0.16}),
    ]

    checked = check_score_consistency(results)

    assert [(item.status, item.error_category) for item in checked] == [
        ("invalid", "RuntimeError"),
        ("ok", None),
        ("ok", None),
        ("ok", None),
    ]
    assert checked[1].max_abs_score_delta == 0.0
    assert checked[2].max_abs_score_delta == pytest.approx(5e-4)
    assert checked[3].max_abs_score_delta == pytest.approx(8e-2)


def test_score_check_rejects_a_level_that_reorders_candidates():
    results = [
        LevelResult(_measurement(1), {"q\x1fa": 0.51, "q\x1fb": 0.49}),
        LevelResult(_measurement(2), {"q\x1fa": 0.49, "q\x1fb": 0.51}),
    ]

    checked = check_score_consistency(results)

    assert [(item.status, item.error_category) for item in checked] == [
        ("ok", None),
        ("invalid", "rank_mismatch"),
    ]
    assert checked[1].max_abs_score_delta == pytest.approx(2e-2)


def test_score_check_rejects_levels_that_scored_different_pairs():
    checked = check_score_consistency(
        [
            LevelResult(_measurement(0), {"q\x1fa": 0.5}),
            LevelResult(_measurement(1), {"q\x1fb": 0.5}),
        ]
    )

    assert (
        checked[1].status,
        checked[1].error_category,
        checked[1].max_abs_score_delta,
    ) == ("invalid", "score_mismatch", None)


def test_percentile_interpolates_between_ranks():
    assert percentile([], 0.95) is None
    assert percentile([2.0], 0.95) == 2.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)


def test_measurement_payload_round_trips_the_full_candidate():
    measurement = BenchmarkMeasurement.invalid(
        LEVELS[2], "ModelServerExited", "x" * 20_000
    )

    payload = measurement.to_dict()

    assert payload["candidate"] == LEVELS[2].to_dict()
    assert BenchmarkMeasurement.from_dict(payload) == measurement
    assert measurement.log_tail is not None
    assert len(measurement.log_tail) == 16_000
    assert math.isinf(measurement.items_per_second)


def test_describe_levels_lists_status_error_and_log_tail():
    text = describe_levels(
        [
            BenchmarkMeasurement.invalid(
                LEVELS[0], "ModelServerExited", "CUDA out of memory"
            )
        ]
    )

    assert '"physical_batch_size": 8192' in text
    assert "status=invalid error=ModelServerExited" in text
    assert "CUDA out of memory" in text
