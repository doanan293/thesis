"""Blind calibration of the judge against a stronger grader (spec §8)."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import spearmanr

from pharma_lab.e2e.configs import E2EConfig
from pharma_lab.e2e.golden import Category, GoldenItem
from pharma_lab.e2e.harness import ANSWERS_FILE, config_dir
from pharma_lab.e2e.judging.code_metrics import cited_sentences, context_blocks
from pharma_lab.e2e.judging.service import JUDGMENTS_FILE
from pharma_lab.e2e.records import AnswerRecord, JsonlStore, Judgement
from pharma_lab.evaluation.artifact_contracts import write_json

CALIBRATION_DIR = "calibration"
ITEMS_FILE = "items.jsonl"
KEY_FILE = "key.json"
GRADES_FILE = "grades.jsonl"
AGREEMENT_FILE = "agreement.json"
CONFIGS = (E2EConfig.FULL, E2EConfig.ONE_STEP)
GROUNDED_PER_CONFIG = 35
SPECIAL_PER_CONFIG = 15
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
    citations: list[int]


class Grade(BaseModel):
    model_config = ConfigDict(extra="forbid")

    blind_id: str
    grader: str
    key_fact_verdicts: list[str] | None = None
    faithfulness: float | None = Field(default=None, ge=0, le=1)
    citation_checks: dict[int, bool] = Field(default_factory=dict)
    injection_followed: bool | None = None


def calibration_dir(run_root: Path) -> Path:
    return Path(run_root) / CALIBRATION_DIR


def _pick(rng: random.Random, pool: Sequence[str], count: int, what: str) -> list[str]:
    if len(pool) < count:
        raise ValueError(f"need {count} {what} answers, found {len(pool)}")
    return rng.sample(sorted(pool), count)


def export_calibration(
    run_root: Path, items: dict[str, GoldenItem], *, seed: int = 0
) -> Path:
    """Write the blind grading file and the key that maps it back."""
    rng = random.Random(seed)
    blind: list[tuple[E2EConfig, AnswerRecord]] = []
    for config in CONFIGS:
        answers = JsonlStore(
            config_dir(run_root, config) / ANSWERS_FILE, AnswerRecord
        ).latest()
        grounded = [
            key
            for key, record in answers.items()
            if items[key].category in (Category.ANSWERABLE, Category.MULTI_TURN)
            and record.answer_mode == "grounded"
        ]
        special = [
            key
            for key in answers
            if items[key].category not in (Category.ANSWERABLE, Category.MULTI_TURN)
        ]
        chosen = _pick(rng, grounded, GROUNDED_PER_CONFIG, f"{config} grounded")
        chosen += _pick(rng, special, SPECIAL_PER_CONFIG, f"{config} special")
        blind.extend((config, answers[key]) for key in chosen)
    rng.shuffle(blind)
    directory = calibration_dir(run_root)
    directory.mkdir(parents=True, exist_ok=True)
    key: dict[str, dict[str, str]] = {}
    lines: list[str] = []
    for number, (config, record) in enumerate(blind, start=1):
        blind_id = f"cal-{number:03d}"
        item = items[record.item_id]
        key[blind_id] = {"config": str(config), "item_id": record.item_id}
        lines.append(
            BlindItem(
                blind_id=blind_id,
                category=item.category.value,
                turns=[turn.model_dump() for turn in item.turns],
                key_facts=[fact.fact for fact in item.reference.key_facts],
                context_text=record.context_text,
                answer_text=record.answer_text,
                citations=sorted(
                    number
                    for number in cited_sentences(record.answer_text)
                    if number in context_blocks(record.context_text)
                ),
            ).model_dump_json()
        )
    path = directory / ITEMS_FILE
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    write_json(directory / KEY_FILE, key)
    return path


def cohen_kappa(a: Sequence[object], b: Sequence[object]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("kappa needs two equally long, non-empty label lists")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
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
    return values[int(0.025 * (len(values) - 1))], values[
        int(0.975 * (len(values) - 1))
    ]


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
) -> AgreementRow:
    if len(pairs) < 2:
        return AgreementRow(
            metric, statistic, len(pairs), math.nan, math.nan, math.nan, False
        )
    value = function([x for x, _ in pairs], [y for _, y in pairs])
    low, high = bootstrap_interval(pairs, function)
    return AgreementRow(
        metric,
        statistic,
        len(pairs),
        value,
        low,
        high,
        not math.isnan(value) and value >= KAPPA_THRESHOLD,
    )


def _floats(a: list[object], b: list[object]) -> float:
    return spearman([float(str(x)) for x in a], [float(str(y)) for y in b])


def _finite(row: AgreementRow) -> dict[str, object]:
    """The row as JSON-safe values: an undefined statistic becomes null."""
    return {
        key: None if isinstance(value, float) and math.isnan(value) else value
        for key, value in asdict(row).items()
    }


def score_calibration(run_root: Path) -> list[AgreementRow]:
    directory = calibration_dir(run_root)
    key = json.loads((directory / KEY_FILE).read_text(encoding="utf-8"))
    grades = {
        grade.blind_id: grade
        for grade in (
            Grade.model_validate_json(line)
            for line in (directory / GRADES_FILE)
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
    }
    judgments = {
        config: JsonlStore(
            config_dir(run_root, config) / JUDGMENTS_FILE, Judgement
        ).latest()
        for config in {entry["config"] for entry in key.values()}
    }
    facts: list[tuple[object, object]] = []
    recall: list[tuple[object, object]] = []
    faithfulness: list[tuple[object, object]] = []
    citations: list[tuple[object, object]] = []
    injection: list[tuple[object, object]] = []
    for blind_id, grade in sorted(grades.items()):
        entry = key[blind_id]
        judged = judgments[entry["config"]].get(entry["item_id"])
        if judged is None:
            raise ValueError(f"{blind_id}: {entry['item_id']} has no judgement")
        if grade.key_fact_verdicts is not None and judged.key_fact_verdicts is not None:
            facts.extend(
                (x == "supported", y == "supported")
                for x, y in zip(
                    judged.key_fact_verdicts, grade.key_fact_verdicts, strict=True
                )
            )
            recall.append(
                (
                    judged.key_fact_recall,
                    grade.key_fact_verdicts.count("supported")
                    / len(grade.key_fact_verdicts),
                )
            )
        if grade.faithfulness is not None and judged.faithfulness is not None:
            faithfulness.append((judged.faithfulness, grade.faithfulness))
        if grade.citation_checks and judged.citation_support is not None:
            checks = list(grade.citation_checks.values())
            citations.append((judged.citation_support, sum(checks) / len(checks)))
        if (
            grade.injection_followed is not None
            and judged.injection_followed is not None
        ):
            injection.append((judged.injection_followed, grade.injection_followed))
    rows = [
        _row("key_fact_supported", "cohen_kappa", facts, cohen_kappa),
        _row("injection_followed", "cohen_kappa", injection, cohen_kappa),
        _row("key_fact_recall", "spearman", recall, _floats),
        _row("faithfulness", "spearman", faithfulness, _floats),
        _row("citation_support", "spearman", citations, _floats),
    ]
    write_json(
        directory / AGREEMENT_FILE,
        {"rows": [_finite(row) for row in rows]},
    )
    return rows
