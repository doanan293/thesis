"""Paired bootstrap comparison of two rerankers on per-query metrics (spec 4.7)."""

from __future__ import annotations

import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from seed_pipeline.artifacts.bundle import load_bundle
from seed_pipeline.artifacts.jsonl import iter_jsonl_objects
from seed_pipeline.config.defaults import DEFAULT_TOP_K
from seed_pipeline.evaluation.artifact_contracts import ArtifactContractError
from seed_pipeline.evaluation.metrics_artifacts import report_dir

# Both reports must score the same evaluation rows over the same candidates.
MATCHING_REPORT_FIELDS = ("evaluation_sha256", "candidate_data_sha256")


@dataclass(frozen=True)
class BootstrapResult:
    mean_difference: float
    ci_low: float
    ci_high: float
    resamples: int


def paired_bootstrap(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    resamples: int,
    seed: int,
) -> BootstrapResult:
    """95% percentile interval of mean(candidate - baseline) over resampled queries."""
    if len(baseline) != len(candidate):
        raise ValueError(
            "paired bootstrap needs one baseline value per candidate value "
            f"({len(baseline)} != {len(candidate)})"
        )
    if not baseline:
        raise ValueError("paired bootstrap needs at least one query")
    if resamples < 2:
        raise ValueError("--resamples must be >= 2")
    differences = [
        after - before for before, after in zip(baseline, candidate, strict=True)
    ]
    size = len(differences)
    rng = random.Random(seed)
    means = [
        statistics.fmean(rng.choices(differences, k=size)) for _ in range(resamples)
    ]
    # n=40 cut points sit at 2.5%, 5%, ..., 97.5%: the first and last bound the 95% CI.
    cuts = statistics.quantiles(means, n=40, method="inclusive")
    return BootstrapResult(statistics.fmean(differences), cuts[0], cuts[-1], resamples)


@dataclass(frozen=True)
class MetricComparisonRequest:
    run_root: Path
    baseline_model: str
    candidate_model: str
    metric: str = "mrr"
    top_k: int = DEFAULT_TOP_K
    window_size: int = 3
    resamples: int = 10_000
    seed: int = 0


@dataclass(frozen=True)
class MetricComparison:
    metric: str
    query_count: int
    baseline_mean: float
    candidate_mean: float
    bootstrap: BootstrapResult

    @property
    def candidate_wins(self) -> bool:
        """Spec 4.7 decision rule: adopt the candidate only if the CI is above 0."""
        return self.bootstrap.ci_low > 0


def load_query_metric(
    report: Path, metric: str
) -> tuple[dict[str, Any], dict[str, float]]:
    bundle = load_bundle(report, expected_type="metrics_report", require_complete=True)
    values: dict[str, float] = {}
    for row in iter_jsonl_objects(bundle.data_path):
        if metric not in row:
            raise ArtifactContractError(
                f"Metric {metric} is missing for query {row.get('query_id')} "
                f"in {bundle.data_path}"
            )
        values[str(row["query_id"])] = float(row[metric])
    return dict(bundle.manifest.identity), values


def run_metric_comparison(request: MetricComparisonRequest) -> MetricComparison:
    if request.baseline_model == request.candidate_model:
        raise ValueError("--baseline and --candidate must name different rerankers")
    baseline_report = report_dir(
        request.run_root,
        top_k=request.top_k,
        window_size=request.window_size,
        model=request.baseline_model,
    )
    candidate_report = report_dir(
        request.run_root,
        top_k=request.top_k,
        window_size=request.window_size,
        model=request.candidate_model,
    )
    baseline_identity, baseline = load_query_metric(baseline_report, request.metric)
    candidate_identity, candidate = load_query_metric(candidate_report, request.metric)
    for name in MATCHING_REPORT_FIELDS:
        if baseline_identity.get(name) != candidate_identity.get(name):
            raise ArtifactContractError(
                f"Reports {baseline_report} and {candidate_report} differ in {name}; "
                "run seed metrics for both rerankers on the same run"
            )
    if baseline.keys() != candidate.keys():
        raise ArtifactContractError(
            f"Reports {baseline_report} and {candidate_report} cover different queries"
        )
    query_ids = sorted(baseline)
    before = [baseline[query_id] for query_id in query_ids]
    after = [candidate[query_id] for query_id in query_ids]
    bootstrap = paired_bootstrap(
        before, after, resamples=request.resamples, seed=request.seed
    )
    return MetricComparison(
        request.metric,
        len(query_ids),
        statistics.fmean(before),
        statistics.fmean(after),
        bootstrap,
    )
