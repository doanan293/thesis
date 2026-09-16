"""Tables for the paper from the stored answers and judgments (spec §9)."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pharma_lab.e2e.calibration import AGREEMENT_FILE, calibration_dir
from pharma_lab.e2e.configs import E2EConfig
from pharma_lab.e2e.golden import ExpectedBehavior, GoldenItem
from pharma_lab.e2e.harness import ANSWERS_FILE, config_dir
from pharma_lab.e2e.judging.service import JUDGMENTS_FILE
from pharma_lab.e2e.records import AnswerRecord, JsonlStore, Judgement

REPORTS_DIR = "reports"
RESAMPLES = 10_000
SEED = 0
ERROR_ROWS = 20

Values = dict[str, float]


def _flag(value: bool | None) -> float | None:
    return None if value is None else float(value)


# metric -> value for one item, or None when the metric does not apply
METRICS: dict[str, Callable[[AnswerRecord, Judgement], float | None]] = {
    "behaviour_correct": lambda _, j: float(j.behaviour_correct),
    "key_fact_recall": lambda _, j: j.key_fact_recall,
    "contradiction": lambda _, j: _flag(j.contradiction),
    "faithfulness": lambda _, j: j.faithfulness,
    "factual_correctness": lambda _, j: j.factual_correctness,
    "answer_relevancy": lambda _, j: j.answer_relevancy,
    "citation_precision": lambda _, j: j.citation_precision,
    "citation_recall": lambda _, j: j.citation_recall,
    "citation_support": lambda _, j: j.citation_support,
    "injection_followed": lambda _, j: _flag(j.injection_followed),
    "declined": lambda _, j: _flag(j.declined),
    "tokens_per_turn": lambda a, _: float(a.total_tokens),
    "llm_calls": lambda a, _: float(a.llm_calls),
    "latency_seconds": lambda a, _: a.latency_seconds,
}
LATENCY = "latency_seconds"


@dataclass(frozen=True)
class Summary:
    n: int
    mean: float
    ci_low: float
    ci_high: float


def bootstrap_mean(values: Sequence[float], *, seed: int = SEED) -> Summary:
    """Mean with a 95% percentile bootstrap interval."""
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return Summary(0, math.nan, math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, data.size, (RESAMPLES, data.size))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return Summary(data.size, float(data.mean()), float(low), float(high))


def paired_delta(baseline: Values, candidate: Values, *, seed: int = SEED) -> Summary:
    """Mean of candidate - baseline over shared items, with a bootstrap interval."""
    shared = sorted(set(baseline) & set(candidate))
    return bootstrap_mean([candidate[key] - baseline[key] for key in shared], seed=seed)


def _values(
    answers: dict[str, AnswerRecord], judgments: dict[str, Judgement]
) -> dict[str, Values]:
    table: dict[str, Values] = {metric: {} for metric in METRICS}
    for key, judgement in judgments.items():
        record = answers.get(key)
        if record is None or judgement.error is not None:
            continue
        for metric, read in METRICS.items():
            value = read(record, judgement)
            if value is not None and not math.isnan(value):
                table[metric][key] = value
    return table


def _load(
    run_root: Path,
) -> dict[str, tuple[dict[str, AnswerRecord], dict[str, Judgement]]]:
    loaded = {}
    for config in E2EConfig:
        directory = config_dir(run_root, config)
        if (directory / JUDGMENTS_FILE).is_file():
            loaded[config.value] = (
                JsonlStore(directory / ANSWERS_FILE, AnswerRecord).latest(),
                JsonlStore(directory / JUDGMENTS_FILE, Judgement).latest(),
            )
    if not loaded:
        raise ValueError(f"no judged configuration under {run_root}")
    return loaded


def _cell(summary: Summary) -> str:
    if summary.n == 0:
        return "--"
    return f"{summary.mean:.3f} [{summary.ci_low:.3f}, {summary.ci_high:.3f}]"


def _write_csv(
    path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _latex(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def _write_tex(
    path: Path, caption: str, header: Sequence[str], rows: Sequence[Sequence[str]]
) -> None:
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{_latex(caption)}}}",
        r"\begin{tabular}{l" + "c" * (len(header) - 1) + "}",
        r"\hline",
        " & ".join(_latex(cell) for cell in header) + r" \\",
        r"\hline",
        *(" & ".join(_latex(cell) for cell in row) + r" \\" for row in rows),
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _percentile(values: Values, q: float) -> float:
    return float(np.percentile(list(values.values()), q)) if values else math.nan


def _section(chunk_label: str) -> str:
    return chunk_label.rsplit(":chunk-", 1)[0]


def error_label(item: GoldenItem, record: AnswerRecord) -> str:
    """Where a poor grounded answer went wrong first."""
    gold = set(item.gold_section_ids)
    if not gold & {_section(label) for label in record.retrieved_chunk_ids}:
        return "retrieval"
    if not gold & {c.section_id for c in record.citations if c.section_id}:
        return "citation"
    return "answer"


def write_report(run_root: Path, items: dict[str, GoldenItem]) -> list[Path]:
    loaded = _load(run_root)
    directory = Path(run_root) / REPORTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    values = {config: _values(*pair) for config, pair in loaded.items()}
    configs = list(loaded)
    written: list[Path] = []

    main_rows: list[list[object]] = []
    tex_rows: list[list[str]] = []
    for metric in METRICS:
        tex_row = [metric]
        for config in configs:
            summary = bootstrap_mean(list(values[config][metric].values()))
            main_rows.append(
                [
                    config,
                    metric,
                    summary.n,
                    summary.mean,
                    summary.ci_low,
                    summary.ci_high,
                ]
            )
            tex_row.append(_cell(summary))
        tex_rows.append(tex_row)
    for q in (50, 95):
        tex_rows.append(
            [f"latency_p{q}"]
            + [f"{_percentile(values[config][LATENCY], q):.2f}" for config in configs]
        )
        main_rows.extend(
            [
                config,
                f"latency_p{q}",
                len(values[config][LATENCY]),
                _percentile(values[config][LATENCY], q),
                "",
                "",
            ]
            for config in configs
        )
    written.append(directory / "main.csv")
    _write_csv(
        written[-1], ["config", "metric", "n", "mean", "ci_low", "ci_high"], main_rows
    )
    written.append(directory / "main.tex")
    _write_tex(
        written[-1],
        "End-to-end results (mean [95% CI])",
        ["metric", *configs],
        tex_rows,
    )

    if E2EConfig.FULL.value in values:
        full = values[E2EConfig.FULL.value]
        others = [config for config in configs if config != E2EConfig.FULL.value]
        ablation_rows: list[list[object]] = []
        ablation_tex: list[list[str]] = []
        for metric in METRICS:
            tex_row = [metric]
            for config in others:
                delta = paired_delta(full[metric], values[config][metric])
                ablation_rows.append(
                    [config, metric, delta.n, delta.mean, delta.ci_low, delta.ci_high]
                )
                tex_row.append(_cell(delta))
            ablation_tex.append(tex_row)
        written.append(directory / "ablation.csv")
        _write_csv(
            written[-1],
            ["config", "metric", "n", "delta", "ci_low", "ci_high"],
            ablation_rows,
        )
        written.append(directory / "ablation.tex")
        _write_tex(
            written[-1],
            "Change against the full agent (paired, mean [95% CI])",
            ["metric", *others],
            ablation_tex,
        )

    group_rows: list[list[object]] = []
    for config in configs:
        for kind in ("category", "eval_group"):
            groups: dict[str, set[str]] = {}
            for key in loaded[config][1]:
                item = items[key]
                name = item.category.value if kind == "category" else item.eval_group
                if name:
                    groups.setdefault(name, set()).add(key)
            for name, keys in sorted(groups.items()):
                for metric in METRICS:
                    selected = [
                        v for k, v in values[config][metric].items() if k in keys
                    ]
                    if selected:
                        group_rows.append(
                            [
                                config,
                                kind,
                                name,
                                metric,
                                len(selected),
                                float(np.mean(selected)),
                            ]
                        )
    written.append(directory / "by_group.csv")
    _write_csv(
        written[-1],
        ["config", "group_type", "group", "metric", "n", "mean"],
        group_rows,
    )

    agreement = calibration_dir(run_root) / AGREEMENT_FILE
    if agreement.is_file():
        rows = json.loads(agreement.read_text(encoding="utf-8"))["rows"]
        header = ["metric", "statistic", "n", "value", "ci_low", "ci_high", "reliable"]
        written.append(directory / "calibration.csv")
        _write_csv(written[-1], header, [[row[h] for h in header] for row in rows])
        written.append(directory / "calibration.tex")
        _write_tex(
            written[-1],
            "Agreement between the judge and the calibration grader",
            ["metric", "statistic", "n", "value [95% CI]"],
            [
                [
                    row["metric"],
                    row["statistic"],
                    str(row["n"]),
                    "--"
                    if row["value"] is None
                    else f"{row['value']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}]",
                ]
                for row in rows
            ],
        )

    if E2EConfig.FULL.value in loaded:
        answers, judgments = loaded[E2EConfig.FULL.value]
        grounded = [
            judgement
            for key, judgement in judgments.items()
            if items[key].expected_behavior is ExpectedBehavior.GROUNDED
            and judgement.key_fact_recall is not None
        ]
        worst = sorted(
            grounded,
            key=lambda j: (
                j.key_fact_recall or 0.0,
                j.faithfulness if j.faithfulness is not None else 1.0,
                j.item_id,
            ),
        )[:ERROR_ROWS]
        lines = [
            "# Lowest-scoring answers of the full agent",
            "",
            "| item | category | key-fact recall | faithfulness | error at | question |",
            "|---|---|---|---|---|---|",
        ]
        for judgement in worst:
            item = items[judgement.item_id]
            faithfulness = (
                "--"
                if judgement.faithfulness is None
                else f"{judgement.faithfulness:.2f}"
            )
            question = item.question.replace("|", "/").replace("\n", " ")
            lines.append(
                f"| {item.item_id} | {item.category.value} | {judgement.key_fact_recall:.2f} "
                f"| {faithfulness} | {error_label(item, answers[judgement.item_id])} | {question} |"
            )
        written.append(directory / "errors.md")
        written[-1].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return written
