from seed_pipeline.runtime.benchmarking import (
    BenchmarkLevel,
    BenchmarkMeasurement,
    compare_cache_arms,
    embedding_levels,
    recommend,
    stratified_sample,
)
from seed_pipeline.runtime.model_profiles import EmbeddingWorkloadProfile


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


def test_embedding_levels_are_cartesian_product():
    profile = EmbeddingWorkloadProfile(8, 1, (4, 8), (1, 2))
    assert embedding_levels(profile) == (
        BenchmarkLevel(4, 1),
        BenchmarkLevel(4, 2),
        BenchmarkLevel(8, 1),
        BenchmarkLevel(8, 2),
    )


def test_recommendation_uses_two_percent_lower_cost_tie_break():
    result = recommend(
        [
            BenchmarkMeasurement(BenchmarkLevel(concurrency=8), 1, 100, 1.0),
            BenchmarkMeasurement(BenchmarkLevel(concurrency=4), 1, 99, 1.0),
        ]
    )
    assert result == BenchmarkLevel(concurrency=4)


def test_recommendation_excludes_disabled_cache_control_arm():
    result = recommend(
        [
            BenchmarkMeasurement(
                BenchmarkLevel(concurrency=1), 1, 100, 1.0, cache_prompt=False
            ),
            BenchmarkMeasurement(
                BenchmarkLevel(concurrency=2), 1, 90, 1.0, cache_prompt=True
            ),
        ]
    )
    assert result == BenchmarkLevel(concurrency=2)


def test_cache_comparison_applies_score_and_performance_gates():
    enabled = BenchmarkMeasurement(
        BenchmarkLevel(), 10, 100, 1.0, latency_p95_seconds=1.0
    )
    disabled = BenchmarkMeasurement(
        BenchmarkLevel(), 10, 100, 1.0, latency_p95_seconds=1.0
    )
    comparison = compare_cache_arms(
        enabled, disabled, max_abs_score_delta=1e-5, top10_agreement=1.0
    )
    assert comparison.accepted is True
