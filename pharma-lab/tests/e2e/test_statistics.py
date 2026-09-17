import math

import pytest

from pharma_lab.e2e.statistics import (
    bootstrap_mean,
    gwet_ac1,
    holm,
    paired_test,
    percent_agreement,
    ppi_mean,
    wilson,
)


def test_wilson_interval_matches_the_textbook_value() -> None:
    # 10/10: the 95% Wilson interval is [0.7225, 1.0].
    summary = wilson([1.0] * 10)
    assert summary.mean == 1.0
    assert summary.ci_low == pytest.approx(0.7225, abs=1e-4)
    assert summary.ci_high == pytest.approx(1.0)
    half = wilson([1.0] * 5 + [0.0] * 5)
    assert (half.ci_low, half.ci_high) == pytest.approx((0.2366, 0.7634), abs=1e-4)
    assert wilson([]).n == 0


def test_bootstrap_mean_of_a_constant_is_exact() -> None:
    summary = bootstrap_mean([2.0, 2.0, 2.0])
    assert (summary.mean, summary.ci_low, summary.ci_high) == (2.0, 2.0, 2.0)


def test_paired_test_separates_a_clear_shift_from_noise() -> None:
    shifted = paired_test([0.5] * 30)
    noise = paired_test([1.0, -1.0] * 15)
    assert shifted.p_value < 0.001
    assert shifted.summary.mean == pytest.approx(0.5)
    assert noise.p_value > 0.5
    assert math.isnan(paired_test([]).p_value)


def test_holm_adjusts_in_input_order() -> None:
    assert holm([0.04, 0.01, 0.03]) == pytest.approx([0.06, 0.03, 0.06])
    adjusted = holm([0.5, math.nan])
    assert adjusted[0] == 0.5 and math.isnan(adjusted[1])


def test_ppi_removes_a_constant_judge_bias() -> None:
    judge_all = [0.9, 0.8, 1.0, 0.7] * 25
    judge_labelled = [0.9, 0.8, 1.0, 0.7]
    human_labelled = [0.7, 0.6, 0.8, 0.5]

    estimate = ppi_mean(judge_all, judge_labelled, human_labelled)

    assert estimate.mean == pytest.approx(0.65)
    assert estimate.ci_low < 0.65 < estimate.ci_high
    with pytest.raises(ValueError, match="one human label"):
        ppi_mean(judge_all, [0.1], [])


def test_gwet_ac1_stays_high_when_kappa_collapses_on_skewed_labels() -> None:
    judge = [True] * 19 + [False]
    human = [True] * 20
    assert percent_agreement(judge, human) == 0.95
    assert gwet_ac1(judge, human) == pytest.approx(0.9474, abs=1e-3)
    assert gwet_ac1([True, True], [True, True]) == 1.0
