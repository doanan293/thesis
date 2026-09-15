from seed_pipeline.runtime.benchmarking import (
    BenchmarkLevel,
    BenchmarkMeasurement,
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
