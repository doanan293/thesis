"""Interval estimates and tests for the E2E tables.

- Proportions use Wilson score intervals, which stay valid for the 10-item
  categories.
- Means use percentile bootstrap intervals, as in the base retrieval paper.
- Configuration differences use a paired randomization (sign-flip) test, as
  recommended for IR evaluation (Smucker et al., CIKM 2007), with Holm's correction
  across the ablations.
- Judge-based means are corrected with prediction-powered inference (Angelopoulos
  et al., Science 2023; ARES, NAACL 2024) using the calibration grades.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

RESAMPLES = 10_000
SEED = 0
Z95 = 1.959963984540054


@dataclass(frozen=True)
class Summary:
    n: int
    mean: float
    ci_low: float
    ci_high: float


def _empty() -> Summary:
    return Summary(0, math.nan, math.nan, math.nan)


def wilson(values: Sequence[float]) -> Summary:
    """Share of ones with a 95% Wilson score interval."""
    n = len(values)
    if n == 0:
        return _empty()
    p = float(np.mean(values))
    denominator = 1 + Z95**2 / n
    centre = (p + Z95**2 / (2 * n)) / denominator
    half = Z95 * math.sqrt(p * (1 - p) / n + Z95**2 / (4 * n * n)) / denominator
    return Summary(n, p, max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_mean(values: Sequence[float], *, seed: int = SEED) -> Summary:
    """Mean with a 95% percentile bootstrap interval."""
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return _empty()
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, data.size, (RESAMPLES, data.size))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return Summary(data.size, float(data.mean()), float(low), float(high))


@dataclass(frozen=True)
class PairedTest:
    summary: Summary
    p_value: float


def paired_test(differences: Sequence[float], *, seed: int = SEED) -> PairedTest:
    """Mean paired difference, its bootstrap interval and a two-sided sign-flip p."""
    data = np.asarray(differences, dtype=float)
    summary = bootstrap_mean(differences, seed=seed)
    if data.size == 0:
        return PairedTest(summary, math.nan)
    observed = abs(data.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(RESAMPLES, data.size))
    permuted = np.abs((signs * data).mean(axis=1))
    extreme = int(np.count_nonzero(permuted >= observed - 1e-12))
    return PairedTest(summary, (extreme + 1) / (RESAMPLES + 1))


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order (NaN stays NaN)."""
    indexed = [(p, i) for i, p in enumerate(p_values) if not math.isnan(p)]
    indexed.sort()
    m = len(indexed)
    adjusted = [math.nan] * len(p_values)
    running = 0.0
    for rank, (p, index) in enumerate(indexed):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[index] = running
    return adjusted


def ppi_mean(
    judge_all: Sequence[float],
    judge_labelled: Sequence[float],
    human_labelled: Sequence[float],
) -> Summary:
    """Prediction-powered estimate of the human mean with a 95% normal interval.

    theta = mean(judge on all items) - mean(judge - human on the labelled sample);
    the labelled sample must be a random draw from the same items.
    """
    if len(judge_labelled) != len(human_labelled):
        raise ValueError("PPI needs one human label per labelled judge score")
    n, big_n = len(human_labelled), len(judge_all)
    if n < 2 or big_n < 2:
        return _empty()
    rectifier = np.asarray(judge_labelled, float) - np.asarray(human_labelled, float)
    estimate = float(np.mean(judge_all) - rectifier.mean())
    variance = (
        float(np.var(judge_all, ddof=1)) / big_n + float(np.var(rectifier, ddof=1)) / n
    )
    half = Z95 * math.sqrt(variance)
    return Summary(n, estimate, estimate - half, estimate + half)


def percent_agreement(a: Sequence[object], b: Sequence[object]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("agreement needs two equally long, non-empty label lists")
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a)


def gwet_ac1(a: Sequence[object], b: Sequence[object]) -> float:
    """Gwet's AC1, robust to skewed label distributions (unlike Cohen's kappa)."""
    observed = percent_agreement(a, b)
    labels = sorted(set(a) | set(b), key=str)
    if len(labels) < 2:
        return 1.0
    n = len(a)
    shares = [
        (list(a).count(label) + list(b).count(label)) / (2 * n) for label in labels
    ]
    expected = sum(share * (1 - share) for share in shares) / (len(labels) - 1)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)
