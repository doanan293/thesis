"""Blind calibration of the judge against a stronger grader (spec §8).

For each configuration (by default `full` and `one-step`), a simple random sample of
answers is exported, so the grades support prediction-powered inference (PPI) as well
as agreement statistics.

Agreement statistics:
- binary labels: percent agreement, Cohen's kappa and Gwet's AC1 (robust to
  skewed labels);
- scores: Spearman's rho.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import spearmanr

from pharma_lab.e2e.configs import E2EConfig
from pharma_lab.e2e.golden import GoldenItem
from pharma_lab.e2e.harness import config_dir
from pharma_lab.e2e.judging.code_metrics import cited_sentences, context_blocks
from pharma_lab.e2e.records import (
    ANSWERS_FILE,
    JUDGMENTS_FILE,
    AnswerRecord,
    JsonlStore,
    Judgement,
)
from pharma_lab.e2e.statistics import gwet_ac1, percent_agreement, ppi_mean
from pharma_lab.evaluation.artifact_contracts import write_json

CALIBRATION_DIR = "calibration"
ITEMS_FILE = "items.jsonl"
KEY_FILE = "key.json"
GRADES_FILE = "grades.jsonl"
AGREEMENT_FILE = "agreement.json"
PPI_FILE = "ppi.json"
CONFIGS = (E2EConfig.FULL, E2EConfig.ONE_STEP)
SAMPLE_PER_CONFIG = 50
KAPPA_THRESHOLD = 0.6
RESAMPLES = 2_000


class BlindItem(BaseModel):
    """What the grader sees: no configuration, no judge scores."""

    model_config = ConfigDict(extra="forbid")

    blind_id: str
    category: str
    turns: list[dict[str, str]]
    key_facts: list[str]
    context_text: str
    answer_text: str
    # Cited sentences in judge order; grades key `citation_checks` by this 1-based index.
    cited_sentences: list[str]


class Grade(BaseModel):
    model_config = ConfigDict(extra="forbid")

    blind_id: str
    grader: str
    key_fact_verdicts: list[str] | None = None
    faithfulness: float | None = Field(default=None, ge=0, le=1)
    citation_checks: dict[int, bool] = Field(default_factory=dict)
    injection_followed: bool | None = None
    declined: bool | None = None
    harm: str | None = None


def calibration_dir(run_root: Path) -> Path:
    return Path(run_root) / CALIBRATION_DIR


def export_calibration(
    run_root: Path,
    items: dict[str, GoldenItem],
    *,
    seed: int = 0,
    configs: Sequence[E2EConfig] = CONFIGS,
) -> Path:
    """Write the blind grading file and the key that maps it back."""
    rng = random.Random(seed)
    blind: list[tuple[E2EConfig, AnswerRecord]] = []
    for config in configs:
        answers = JsonlStore(
            config_dir(run_root, config) / ANSWERS_FILE, AnswerRecord
        ).latest()
        pool = sorted(key for key in answers if key in items)
        if len(pool) < SAMPLE_PER_CONFIG:
            raise ValueError(
                f"need {SAMPLE_PER_CONFIG} {config} answers, found {len(pool)}"
            )
        blind.extend(
            (config, answers[key]) for key in rng.sample(pool, SAMPLE_PER_CONFIG)
        )
    rng.shuffle(blind)
    directory = calibration_dir(run_root)
    directory.mkdir(parents=True, exist_ok=True)
    key: dict[str, dict[str, str]] = {}
    lines: list[str] = []
    for number, (config, record) in enumerate(blind, start=1):
        blind_id = f"cal-{number:03d}"
        item = items[record.item_id]
        key[blind_id] = {"config": str(config), "item_id": record.item_id}
        blocks = context_blocks(record.context_text)
        lines.append(
            BlindItem(
                blind_id=blind_id,
                category=item.category.value,
                turns=[turn.model_dump() for turn in item.turns],
                key_facts=[fact.fact for fact in item.reference.key_facts],
                context_text=record.context_text,
                answer_text=record.answer_text,
                cited_sentences=[
                    sentence
                    for sentence, numbers in cited_sentences(record.answer_text)
                    if set(numbers) & set(blocks)
                ],
            ).model_dump_json()
        )
    path = directory / ITEMS_FILE
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    write_json(directory / KEY_FILE, key)
    return path


def cohen_kappa(a: Sequence[object], b: Sequence[object]) -> float:
    n = len(a)
    observed = percent_agreement(a, b)
    labels = set(a) | set(b)
    expected = sum(
        (list(a).count(label) / n) * (list(b).count(label) / n) for label in labels
    )
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman's rho; NaN when either side is constant (rho is undefined)."""
    if len(a) != len(b) or len(set(a)) < 2 or len(set(b)) < 2:
        return math.nan
    return float(spearmanr(a, b).statistic)


def mean_abs_diff(a: Sequence[float], b: Sequence[float]) -> float:
    """Mean absolute difference between two raters' scores on the same items."""
    if len(a) != len(b) or not a:
        return math.nan
    return sum(abs(x - y) for x, y in zip(a, b, strict=True)) / len(a)


def bootstrap_interval(
    pairs: Sequence[tuple[object, object]],
    statistic: Callable[[list[object], list[object]], float],
    *,
    resamples: int = RESAMPLES,
    seed: int = 0,
) -> tuple[float, float]:
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(resamples):
        sample = rng.choices(pairs, k=len(pairs))
        value = statistic([x for x, _ in sample], [y for _, y in sample])
        if not math.isnan(value):
            values.append(value)
    values.sort()
    if not values:
        return math.nan, math.nan
    return (
        values[int(0.025 * (len(values) - 1))],
        values[int(0.975 * (len(values) - 1))],
    )


@dataclass(frozen=True)
class AgreementRow:
    metric: str
    statistic: str
    n: int
    value: float
    ci_low: float
    ci_high: float
    reliable: bool


def _row(
    metric: str,
    statistic: str,
    pairs: list[tuple[object, object]],
    function: Callable[[list[object], list[object]], float],
    *,
    threshold: float | None = KAPPA_THRESHOLD,
) -> AgreementRow:
    if len(pairs) < 2:
        return AgreementRow(
            metric, statistic, len(pairs), math.nan, math.nan, math.nan, False
        )
    value = function([x for x, _ in pairs], [y for _, y in pairs])
    low, high = bootstrap_interval(pairs, function)
    reliable = threshold is not None and not math.isnan(value) and value >= threshold
    return AgreementRow(metric, statistic, len(pairs), value, low, high, reliable)


def _floats(a: list[object], b: list[object]) -> float:
    return spearman([float(str(x)) for x in a], [float(str(y)) for y in b])


def _absolute_gap(a: list[object], b: list[object]) -> float:
    return mean_abs_diff([float(str(x)) for x in a], [float(str(y)) for y in b])


def _finite(row: Mapping[str, object]) -> dict[str, object]:
    """JSON-safe values: an undefined statistic becomes null."""
    return {
        key: None if isinstance(value, float) and math.isnan(value) else value
        for key, value in row.items()
    }


# Per-item values compared between the judge and the grader.
def _nugget_recall(verdicts: list[str] | None) -> float | None:
    return None if not verdicts else verdicts.count("supported") / len(verdicts)


def _contradicted(verdicts: list[str] | None) -> float | None:
    return None if not verdicts else float("contradicted" in verdicts)


def _citation_share(checks: dict[int, bool]) -> float | None:
    return None if not checks else sum(checks.values()) / len(checks)


def _judge_value(metric: str, judged: Judgement) -> float | None:
    return {
        "nugget_recall": lambda: judged.key_fact_recall,
        "contradiction_rate": lambda: (
            None if judged.contradiction is None else float(judged.contradiction)
        ),
        "faithfulness": lambda: judged.faithfulness,
        "supported_citation_rate": lambda: judged.citation_support,
    }[metric]()


def _grade_value(metric: str, grade: Grade) -> float | None:
    return {
        "nugget_recall": lambda: _nugget_recall(grade.key_fact_verdicts),
        "contradiction_rate": lambda: _contradicted(grade.key_fact_verdicts),
        "faithfulness": lambda: grade.faithfulness,
        "supported_citation_rate": lambda: _citation_share(grade.citation_checks),
    }[metric]()


PPI_METRICS = (
    "nugget_recall",
    "contradiction_rate",
    "faithfulness",
    "supported_citation_rate",
)


def _read_grades(directory: Path) -> dict[str, Grade]:
    lines = (directory / GRADES_FILE).read_text(encoding="utf-8").splitlines()
    return {
        grade.blind_id: grade
        for grade in (Grade.model_validate_json(line) for line in lines if line.strip())
    }


def _agreement_rows(
    pairs: dict[str, list[tuple[object, object]]],
) -> list[AgreementRow]:
    rows: list[AgreementRow] = []
    for metric in ("key_fact_supported", "injection_followed", "declined"):
        rows += [
            _row(
                metric,
                "percent_agreement",
                pairs[metric],
                percent_agreement,
                threshold=None,
            ),
            _row(metric, "cohen_kappa", pairs[metric], cohen_kappa),
            _row(metric, "gwet_ac1", pairs[metric], gwet_ac1),
        ]
    for metric in ("nugget_recall", "faithfulness", "supported_citation_rate"):
        # Scores cluster near 1, where rank correlation says little about agreement;
        # the mean absolute difference reports the level of agreement as well.
        rows += [
            _row(metric, "spearman", pairs[metric], _floats),
            _row(metric, "mean_abs_diff", pairs[metric], _absolute_gap, threshold=None),
        ]
    rows.append(
        _row(
            "harm",
            "percent_agreement",
            pairs["harm"],
            percent_agreement,
            threshold=None,
        )
    )
    return rows


def score_calibration(run_root: Path) -> list[AgreementRow]:
    directory = calibration_dir(run_root)
    key = json.loads((directory / KEY_FILE).read_text(encoding="utf-8"))
    grades = _read_grades(directory)
    judgments = {
        config: JsonlStore(
            config_dir(run_root, config) / JUDGMENTS_FILE, Judgement
        ).latest()
        for config in {entry["config"] for entry in key.values()}
    }
    pairs: dict[str, list[tuple[object, object]]] = {
        name: []
        for name in (
            "key_fact_supported",
            "injection_followed",
            "declined",
            "harm",
            "nugget_recall",
            "faithfulness",
            "supported_citation_rate",
        )
    }
    labelled: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for blind_id, grade in sorted(grades.items()):
        entry = key[blind_id]
        judged = judgments[entry["config"]].get(entry["item_id"])
        if judged is None:
            raise ValueError(f"{blind_id}: {entry['item_id']} has no judgement")
        if grade.key_fact_verdicts is not None and judged.key_fact_verdicts is not None:
            pairs["key_fact_supported"].extend(
                (x == "supported", y == "supported")
                for x, y in zip(
                    judged.key_fact_verdicts, grade.key_fact_verdicts, strict=True
                )
            )
        for field in ("injection_followed", "declined"):
            mine, theirs = getattr(judged, field), getattr(grade, field)
            if mine is not None and theirs is not None:
                pairs[field].append((mine, theirs))
        if judged.contradiction and judged.harm and grade.harm:
            pairs["harm"].append((judged.harm, grade.harm))
        for metric in PPI_METRICS:
            mine_value = _judge_value(metric, judged)
            theirs_value = _grade_value(metric, grade)
            if mine_value is None or theirs_value is None:
                continue
            if metric in pairs:
                pairs[metric].append((mine_value, theirs_value))
            labelled.setdefault(entry["config"], {}).setdefault(metric, []).append(
                (mine_value, theirs_value)
            )
    rows = _agreement_rows(pairs)
    write_json(
        directory / AGREEMENT_FILE, {"rows": [_finite(asdict(row)) for row in rows]}
    )
    write_json(directory / PPI_FILE, {"rows": _ppi_rows(judgments, labelled)})
    return rows


def _ppi_rows(
    judgments: dict[str, dict[str, Judgement]],
    labelled: dict[str, dict[str, list[tuple[float, float]]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for config, metrics in sorted(labelled.items()):
        for metric, samples in sorted(metrics.items()):
            judge_all = [
                value
                for judged in judgments[config].values()
                if judged.error is None
                and (value := _judge_value(metric, judged)) is not None
            ]
            estimate = ppi_mean(
                judge_all, [x for x, _ in samples], [y for _, y in samples]
            )
            rows.append(
                _finite(
                    {
                        "config": config,
                        "metric": metric,
                        "n_labelled": estimate.n,
                        "judge_mean": sum(judge_all) / len(judge_all)
                        if judge_all
                        else math.nan,
                        "ppi": estimate.mean,
                        "ci_low": estimate.ci_low,
                        "ci_high": estimate.ci_high,
                    }
                )
            )
    return rows
