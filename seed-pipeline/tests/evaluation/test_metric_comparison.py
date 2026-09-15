from dataclasses import replace
from pathlib import Path

import pytest

from seed_pipeline.evaluation.artifact_contracts import ArtifactContractError
from seed_pipeline.evaluation.metric_comparison import (
    BootstrapResult,
    MetricComparisonRequest,
    paired_bootstrap,
    run_metric_comparison,
)
from seed_pipeline.evaluation.metrics_artifacts import publish_metrics_artifact
from seed_pipeline.evaluation.variant_identity import MetricsArtifactIdentity

BASELINE = "qwen3-reranker:0.6b-fp16"
CANDIDATE = "qwen3-reranker:4b-fp16"
BALANCED = [1.0 if index % 2 else -1.0 for index in range(1000)]


def test_a_constant_improvement_has_a_zero_width_interval() -> None:
    result = paired_bootstrap([0.5, 0.25, 0.0], [1.0, 0.75, 0.5], resamples=200, seed=0)

    assert result == BootstrapResult(0.5, 0.5, 0.5, 200)


def test_a_clear_improvement_puts_the_lower_bound_above_zero() -> None:
    result = paired_bootstrap(
        [0.0] * 1000, [0.5] * 900 + [-0.5] * 100, resamples=2000, seed=0
    )

    assert result.mean_difference == pytest.approx(0.4)
    assert 0 < result.ci_low < result.mean_difference < result.ci_high


def test_balanced_differences_keep_zero_inside_the_interval() -> None:
    result = paired_bootstrap([0.0] * 1000, BALANCED, resamples=2000, seed=0)

    assert result.mean_difference == 0.0
    assert result.ci_low < 0 < result.ci_high


def test_the_same_seed_repeats_and_another_seed_resamples_differently() -> None:
    first = paired_bootstrap([0.0] * 1000, BALANCED, resamples=2000, seed=0)

    assert paired_bootstrap([0.0] * 1000, BALANCED, resamples=2000, seed=0) == first
    assert paired_bootstrap([0.0] * 1000, BALANCED, resamples=2000, seed=1) != first


@pytest.mark.parametrize(
    ("baseline", "candidate", "resamples", "message"),
    [
        ([0.1], [0.1, 0.2], 10, "one baseline value per candidate value"),
        ([], [], 10, "at least one query"),
        ([0.1], [0.2], 1, "--resamples must be >= 2"),
    ],
)
def test_invalid_bootstrap_inputs_are_rejected(
    baseline: list[float], candidate: list[float], resamples: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        paired_bootstrap(baseline, candidate, resamples=resamples, seed=0)


def _publish(
    run: Path,
    model: str,
    mrr: dict[str, float],
    *,
    candidate_data_sha256: str = "c" * 64,
) -> None:
    rows = [
        {"query_id": query_id, "hit@10": 1, "mrr": value}
        for query_id, value in mrr.items()
    ]
    publish_metrics_artifact(
        run,
        MetricsArtifactIdentity.create(
            evaluation_sha256="e" * 64,
            candidate_data_sha256=candidate_data_sha256,
            top_k=30,
            window_size=3,
            rerank_variant_sha256=f"{model} variant",
        ),
        {"count": len(rows), "mrr": sum(mrr.values())},
        {"eval_group": {}, "difficulty": {}},
        rows,
        top_k=30,
        window_size=3,
        model=model,
        variant_sha256=f"{model} variant",
    )


def _request(run: Path) -> MetricComparisonRequest:
    return MetricComparisonRequest(
        run, BASELINE, CANDIDATE, metric="mrr", top_k=30, window_size=3, resamples=500
    )


def test_comparison_pairs_the_same_queries_from_both_reports(tmp_path: Path) -> None:
    _publish(tmp_path, BASELINE, {"q1": 0.5, "q2": 0.0, "q3": 0.5, "q4": 0.0})
    _publish(tmp_path, CANDIDATE, {"q4": 1.0, "q3": 1.0, "q2": 1.0, "q1": 1.0})

    comparison = run_metric_comparison(_request(tmp_path))

    assert (comparison.metric, comparison.query_count) == ("mrr", 4)
    assert (comparison.baseline_mean, comparison.candidate_mean) == (0.25, 1.0)
    assert comparison.bootstrap.mean_difference == 0.75
    assert comparison.bootstrap.ci_low >= 0.5
    assert comparison.candidate_wins is True


def test_equal_reports_keep_the_baseline(tmp_path: Path) -> None:
    same = {"q1": 0.5, "q2": 1.0}
    _publish(tmp_path, BASELINE, same)
    _publish(tmp_path, CANDIDATE, same)

    comparison = run_metric_comparison(_request(tmp_path))

    assert comparison.bootstrap == BootstrapResult(0.0, 0.0, 0.0, 500)
    assert comparison.candidate_wins is False


def test_reports_over_different_candidates_are_not_compared(tmp_path: Path) -> None:
    _publish(tmp_path, BASELINE, {"q1": 0.5})
    _publish(tmp_path, CANDIDATE, {"q1": 1.0}, candidate_data_sha256="d" * 64)

    with pytest.raises(ArtifactContractError, match="candidate_data_sha256"):
        run_metric_comparison(_request(tmp_path))


def test_reports_over_different_queries_are_not_compared(tmp_path: Path) -> None:
    _publish(tmp_path, BASELINE, {"q1": 0.5, "q2": 0.5})
    _publish(tmp_path, CANDIDATE, {"q1": 1.0, "q3": 1.0})

    with pytest.raises(ArtifactContractError, match="different queries"):
        run_metric_comparison(_request(tmp_path))


def test_a_missing_metric_and_one_model_twice_are_rejected(tmp_path: Path) -> None:
    _publish(tmp_path, BASELINE, {"q1": 0.5})
    _publish(tmp_path, CANDIDATE, {"q1": 1.0})

    with pytest.raises(ArtifactContractError, match="Metric hit@3 is missing"):
        run_metric_comparison(replace(_request(tmp_path), metric="hit@3"))
    with pytest.raises(ValueError, match="different rerankers"):
        run_metric_comparison(replace(_request(tmp_path), candidate_model=BASELINE))
