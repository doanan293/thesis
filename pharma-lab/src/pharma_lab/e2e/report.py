"""Tables for the paper from the stored answers and judgments (spec §9).

Metric names follow the literature they come from:
- truthfulness and the perfect / acceptable / missing / incorrect classes: CRAG
  (NeurIPS 2024);
- nugget recall: TREC 2024 RAG;
- faithfulness and answer relevancy: RAGAS (EACL 2024);
- citation recall: ALCE (EMNLP 2023);
- negative rejection: RGB (AAAI 2024);
- attack success rate: AgentDojo (NeurIPS 2024);
- harm: CHART / TRIPOD-LLM.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pharma_lab.e2e.calibration import AGREEMENT_FILE, PPI_FILE, calibration_dir
from pharma_lab.e2e.configs import E2EConfig
from pharma_lab.e2e.golden import Category, ExpectedBehavior, GoldenItem
from pharma_lab.e2e.harness import ANSWERS_FILE, config_dir
from pharma_lab.e2e.judging.code_metrics import (
    CRAG_SCORES,
    alce_citation_recall,
    behaviour_correct,
    citation_scores,
    crag_label,
    is_relevant_chunk,
    over_refusal,
)
from pharma_lab.e2e.judging.service import JUDGE_FILE, JUDGMENTS_FILE
from pharma_lab.e2e.records import AnswerRecord, JsonlStore, Judgement
from pharma_lab.e2e.run_identity import RUN_FILE
from pharma_lab.e2e.statistics import (
    RESAMPLES,
    SEED,
    Summary,
    bootstrap_mean,
    holm,
    paired_test,
    wilson,
)
from pharma_lab.evaluation.relevance_judgments import LoadedJudgments

REPORTS_DIR = "reports"
ERROR_ROWS = 20

Values = dict[str, float]


@dataclass(frozen=True)
class ItemView:
    """Everything one metric may read about one item in one configuration."""

    item: GoldenItem
    answer: AnswerRecord
    judgement: Judgement
    # Gold section id -> extra accepted chunk ids (relevance judgments).
    accepted: Mapping[str, frozenset[str]]


@dataclass(frozen=True)
class Metric:
    name: str
    read: Callable[[ItemView], float | None]
    binary: bool
    primary: bool


def _flag(value: bool | None) -> float | None:
    return None if value is None else float(value)


def _crag(view: ItemView) -> str | None:
    return crag_label(
        view.item,
        view.answer,
        nugget_recall=view.judgement.key_fact_recall,
        contradiction=view.judgement.contradiction,
        declined=view.judgement.declined,
    )


def _crag_score(view: ItemView) -> float | None:
    label = _crag(view)
    return None if label is None else CRAG_SCORES[label]


def _crag_is(label: str) -> Callable[[ItemView], float | None]:
    def read(view: ItemView) -> float | None:
        current = _crag(view)
        return None if current is None else float(current == label)

    return read


def _behaviour_in(category: Category) -> Callable[[ItemView], float | None]:
    def read(view: ItemView) -> float | None:
        if view.item.category is not category:
            return None
        return float(
            behaviour_correct(
                view.item,
                view.answer,
                injection_followed=view.judgement.injection_followed,
                declined=view.judgement.declined,
            )
        )

    return read


def _response(view: ItemView) -> float | None:
    refused = over_refusal(view.item, view.answer)
    return None if refused is None else 1.0 - refused


def _harm_at_least(levels: frozenset[str]) -> Callable[[ItemView], float | None]:
    def read(view: ItemView) -> float | None:
        judged = view.judgement
        if judged.key_fact_recall is None or view.answer.answer_mode != "grounded":
            return None
        if judged.contradiction and judged.harm is None:
            return None  # harm not judged yet
        return float(bool(judged.contradiction) and judged.harm in levels)

    return read


def _cited_source(index: int) -> Callable[[ItemView], float | None]:
    return lambda view: citation_scores(view.item, view.answer, view.accepted)[index]


def _alce(view: ItemView) -> float | None:
    if view.answer.answer_mode != "grounded" or view.judgement.key_fact_recall is None:
        return None
    return alce_citation_recall(
        view.answer.answer_text,
        view.answer.context_text,
        view.judgement.citation_support,
    )


METRICS: tuple[Metric, ...] = (
    # Truthfulness (CRAG): wrong answers cost more than missing ones.
    Metric("truthfulness", _crag_score, False, True),
    Metric("perfect_rate", _crag_is("perfect"), True, True),
    Metric("acceptable_rate", _crag_is("acceptable"), True, False),
    Metric("missing_rate", _crag_is("missing"), True, True),
    Metric("hallucination_rate", _crag_is("incorrect"), True, True),
    # Correctness against reference nuggets, and the harm of wrong answers.
    Metric("nugget_recall", lambda v: v.judgement.key_fact_recall, False, True),
    Metric(
        "contradiction_rate", lambda v: _flag(v.judgement.contradiction), True, False
    ),
    Metric("severe_harm_rate", _harm_at_least(frozenset({"severe"})), True, True),
    Metric(
        "harmful_error_rate",
        _harm_at_least(frozenset({"minor", "severe"})),
        True,
        False,
    ),
    # Grounding and attribution.
    Metric("faithfulness", lambda v: v.judgement.faithfulness, False, True),
    Metric("citation_recall", _alce, False, True),
    Metric(
        "supported_citation_rate", lambda v: v.judgement.citation_support, False, False
    ),
    Metric("cited_source_precision", _cited_source(0), False, False),
    Metric("cited_source_recall", _cited_source(1), True, False),
    # Behaviour.
    Metric("response_rate", _response, True, True),
    Metric("over_refusal_rate", lambda v: over_refusal(v.item, v.answer), True, False),
    Metric("negative_rejection_rate", _behaviour_in(Category.UNANSWERABLE), True, True),
    Metric("redirect_accuracy", _behaviour_in(Category.OUT_OF_SCOPE), True, True),
    Metric("injection_defence_rate", _behaviour_in(Category.INJECTION), True, True),
    Metric(
        "attack_success_rate",
        lambda v: _flag(v.judgement.injection_followed),
        True,
        False,
    ),
    # Secondary answer-quality scores.
    Metric("answer_relevancy", lambda v: v.judgement.answer_relevancy, False, False),
    Metric(
        "factual_correctness_f1",
        lambda v: v.judgement.factual_correctness,
        False,
        False,
    ),
    # Cost.
    Metric("tokens_per_turn", lambda v: float(v.answer.total_tokens), False, True),
    Metric("llm_calls_per_turn", lambda v: float(v.answer.llm_calls), False, True),
    Metric("latency_seconds", lambda v: v.answer.latency_seconds, False, False),
)
BY_NAME = {metric.name: metric for metric in METRICS}
LATENCY = "latency_seconds"

# Names printed in the paper's LaTeX tables, with the direction that is better.
_UP, _DOWN = r"$\uparrow$", r"$\downarrow$"
METRIC_LABELS: Mapping[str, str] = {
    "truthfulness": f"Truthfulness {_UP}",
    "perfect_rate": f"Perfect {_UP}",
    "missing_rate": f"Missing {_DOWN}",
    "hallucination_rate": f"Hallucination {_DOWN}",
    "nugget_recall": f"Nugget recall {_UP}",
    "severe_harm_rate": f"Severe harm {_DOWN}",
    "faithfulness": f"Faithfulness {_UP}",
    "citation_recall": f"Citation recall {_UP}",
    "response_rate": f"Response rate {_UP}",
    "negative_rejection_rate": f"Negative rejection {_UP}",
    "redirect_accuracy": f"Redirect accuracy {_UP}",
    "injection_defence_rate": f"Injection defence {_UP}",
    "tokens_per_turn": f"Tokens / turn {_DOWN}",
    "llm_calls_per_turn": f"LLM calls / turn {_DOWN}",
}
CONFIG_LABELS: Mapping[str, str] = {
    E2EConfig.FULL.value: "Full agent",
    E2EConfig.ONE_STEP.value: "One-step RAG",
    E2EConfig.NO_JUDGE_REFINE.value: "w/o judge--refine",
    E2EConfig.NO_REPHRASE.value: "w/o rephrase",
    E2EConfig.NO_RERANK.value: "w/o rerank",
}
# The LNCS text block is 12.2 cm wide: the main table compares the two systems,
# and the ablation table carries the other configurations as paired deltas.
PAPER_MAIN_CONFIGS: tuple[str, ...] = (E2EConfig.FULL.value, E2EConfig.ONE_STEP.value)


def summarise(metric: Metric, values: Sequence[float]) -> Summary:
    return wilson(values) if metric.binary else bootstrap_mean(values)


def paired_delta(baseline: Values, candidate: Values, *, seed: int = SEED) -> Summary:
    """Mean of candidate - baseline over shared items, with a bootstrap interval."""
    shared = sorted(set(baseline) & set(candidate))
    return bootstrap_mean([candidate[key] - baseline[key] for key in shared], seed=seed)


def _accepted(
    item: GoldenItem, relevance: LoadedJudgments | None
) -> Mapping[str, frozenset[str]]:
    if relevance is None or item.source_query_id is None:
        return {}
    return relevance.for_query(item.source_query_id)


def _values(
    items: Mapping[str, GoldenItem],
    answers: dict[str, AnswerRecord],
    judgments: dict[str, Judgement],
    relevance: LoadedJudgments | None,
) -> dict[str, Values]:
    table: dict[str, Values] = {metric.name: {} for metric in METRICS}
    for key, judgement in judgments.items():
        record = answers.get(key)
        if record is None or judgement.error is not None:
            continue
        view = ItemView(items[key], record, judgement, _accepted(items[key], relevance))
        for metric in METRICS:
            value = metric.read(view)
            if value is not None and not math.isnan(value):
                table[metric.name][key] = value
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


def _number(value: float) -> str:
    if abs(value) >= 100:
        text = f"{value:.0f}"
    elif abs(value) >= 1:
        text = f"{value:.1f}"
    else:
        text = f"{value:.3f}"
    return "$-$" + text[1:] if text.startswith("-") else text


def _cell(summary: Summary) -> str:
    if summary.n == 0:
        return "--"
    return (
        f"{_number(summary.mean)} "
        f"[{_number(summary.ci_low)}, {_number(summary.ci_high)}]"
    )


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
    path: Path,
    caption: str,
    label: str,
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> None:
    """A booktabs table; caption, header and cells are already LaTeX."""
    lines = [
        r"\begin{table}[tb]",
        r"\centering",
        rf"\caption{{{caption}}}\label{{{label}}}",
        r"\small",
        r"\begin{tabular}{l" + "r" * (len(header) - 1) + "}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
        *(" & ".join(row) + r" \\" for row in rows),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _percentile(values: Values, q: float) -> float:
    return float(np.percentile(list(values.values()), q)) if values else math.nan


def _section(chunk_label: str) -> str:
    return chunk_label.rsplit(":chunk-", 1)[0]


def error_label(
    item: GoldenItem,
    record: AnswerRecord,
    accepted: Mapping[str, frozenset[str]] | None = None,
) -> str:
    """Where a poor grounded answer went wrong first."""
    judged = accepted or {}
    if not any(
        is_relevant_chunk(item, _section(label), label, judged)
        for label in record.retrieved_chunk_ids
    ):
        return "retrieval"
    if not any(
        is_relevant_chunk(item, c.section_id, c.chunk_id, judged)
        for c in record.citations
    ):
        return "citation"
    return "answer"


def _main_tables(
    directory: Path, configs: Sequence[str], values: Mapping[str, dict[str, Values]]
) -> list[Path]:
    rows: list[list[object]] = []
    tex_rows: list[list[str]] = []
    paper_configs = [config for config in PAPER_MAIN_CONFIGS if config in configs]
    for metric in METRICS:
        tex_row = [METRIC_LABELS.get(metric.name, metric.name)]
        for config in configs:
            summary = summarise(metric, list(values[config][metric.name].values()))
            rows.append(
                [
                    config,
                    metric.name,
                    "wilson" if metric.binary else "bootstrap",
                    metric.primary,
                    summary.n,
                    summary.mean,
                    summary.ci_low,
                    summary.ci_high,
                ]
            )
            if config in paper_configs:
                tex_row.append(_cell(summary))
        if metric.primary:
            tex_rows.append(tex_row)
    for q in (50, 95):
        for config in configs:
            latency = values[config][LATENCY]
            rows.append(
                [
                    config,
                    f"latency_p{q}",
                    "percentile",
                    False,
                    len(latency),
                    _percentile(latency, q),
                    "",
                    "",
                ]
            )
    main_csv = directory / "main.csv"
    _write_csv(
        main_csv,
        ["config", "metric", "interval", "primary", "n", "mean", "ci_low", "ci_high"],
        rows,
    )
    main_tex = directory / "main.tex"
    _write_tex(
        main_tex,
        "End-to-end results of the full agent and one-step RAG: mean [95\\% CI] "
        "(Wilson for rates, percentile bootstrap otherwise).",
        "tab:e2e-main",
        ["Metric", *(CONFIG_LABELS[config] for config in paper_configs)],
        tex_rows,
    )
    return [main_csv, main_tex]


def _ablation_tables(
    directory: Path, configs: Sequence[str], values: Mapping[str, dict[str, Values]]
) -> list[Path]:
    full = values[E2EConfig.FULL.value]
    others = [config for config in configs if config != E2EConfig.FULL.value]
    rows: list[list[object]] = []
    tex_rows: list[list[str]] = []
    for metric in METRICS:
        tests = []
        for config in others:
            shared = sorted(set(full[metric.name]) & set(values[config][metric.name]))
            tests.append(
                paired_test(
                    [
                        values[config][metric.name][key] - full[metric.name][key]
                        for key in shared
                    ]
                )
            )
        adjusted = holm([test.p_value for test in tests])
        tex_row = [METRIC_LABELS.get(metric.name, metric.name)]
        for config, test, p_holm in zip(others, tests, adjusted, strict=True):
            s = test.summary
            rows.append(
                [
                    config,
                    metric.name,
                    s.n,
                    s.mean,
                    s.ci_low,
                    s.ci_high,
                    test.p_value,
                    p_holm,
                ]
            )
            mark = "" if math.isnan(p_holm) or p_holm >= 0.05 else "$^{*}$"
            tex_row.append("--" if s.n == 0 else f"{_number(s.mean)}{mark}")
        if metric.primary:
            tex_rows.append(tex_row)
    ablation_csv = directory / "ablation.csv"
    _write_csv(
        ablation_csv,
        ["config", "metric", "n", "delta", "ci_low", "ci_high", "p_value", "p_holm"],
        rows,
    )
    ablation_tex = directory / "ablation.tex"
    _write_tex(
        ablation_tex,
        "Paired change of each configuration against the full agent. "
        "$^{*}$: Holm-adjusted randomization $p<0.05$; confidence intervals are "
        "in the released CSV.",
        "tab:e2e-ablation",
        ["Metric", *(CONFIG_LABELS[config] for config in others)],
        tex_rows,
    )
    return [ablation_csv, ablation_tex]


def _group_table(
    directory: Path,
    configs: Sequence[str],
    items: Mapping[str, GoldenItem],
    loaded: Mapping[str, tuple[dict[str, AnswerRecord], dict[str, Judgement]]],
    values: Mapping[str, dict[str, Values]],
) -> Path:
    rows: list[list[object]] = []
    for config in configs:
        for kind in ("category", "eval_group"):
            groups: dict[str, set[str]] = {}
            for key in loaded[config][1]:
                item = items[key]
                if kind == "category":
                    name = item.category.value
                else:
                    # Multi-turn items borrow their follow-up row's group; keep the
                    # eval-group breakdown to single-turn answerable questions.
                    name = (
                        item.eval_group
                        if item.category is Category.ANSWERABLE
                        else None
                    )
                if name:
                    groups.setdefault(name, set()).add(key)
            for name, keys in sorted(groups.items()):
                for metric in METRICS:
                    selected = [
                        v for k, v in values[config][metric.name].items() if k in keys
                    ]
                    if selected:
                        s = summarise(metric, selected)
                        rows.append(
                            [
                                config,
                                kind,
                                name,
                                metric.name,
                                s.n,
                                s.mean,
                                s.ci_low,
                                s.ci_high,
                            ]
                        )
    path = directory / "by_group.csv"
    _write_csv(
        path,
        ["config", "group_type", "group", "metric", "n", "mean", "ci_low", "ci_high"],
        rows,
    )
    return path


def _calibration_tables(directory: Path, run_root: Path) -> list[Path]:
    written: list[Path] = []
    agreement = calibration_dir(run_root) / AGREEMENT_FILE
    if agreement.is_file():
        rows = json.loads(agreement.read_text(encoding="utf-8"))["rows"]
        header = [
            "metric",
            "statistic",
            "n",
            "value",
            "ci_low",
            "ci_high",
            "reliable",
        ]
        written.append(directory / "calibration.csv")
        _write_csv(written[-1], header, [[row.get(h) for h in header] for row in rows])
        written.append(directory / "calibration.tex")
        _write_tex(
            written[-1],
            "Agreement between the LLM judge and the blind calibration grader.",
            "tab:e2e-calibration",
            ["Label", "Statistic", "$n$", "Value [95\\% CI]"],
            [
                [
                    _latex(row["metric"]),
                    _latex(row["statistic"]),
                    str(row["n"]),
                    "--"
                    if row["value"] is None
                    else f"{_number(row['value'])} "
                    f"[{_number(row['ci_low'])}, {_number(row['ci_high'])}]",
                ]
                for row in rows
            ],
        )
    ppi = calibration_dir(run_root) / PPI_FILE
    if ppi.is_file():
        rows = json.loads(ppi.read_text(encoding="utf-8"))["rows"]
        header = [
            "config",
            "metric",
            "n_labelled",
            "judge_mean",
            "ppi",
            "ci_low",
            "ci_high",
        ]
        written.append(directory / "ppi.csv")
        _write_csv(written[-1], header, [[row.get(h) for h in header] for row in rows])
    return written


def _error_analysis(
    directory: Path,
    items: Mapping[str, GoldenItem],
    answers: Mapping[str, AnswerRecord],
    judgments: Mapping[str, Judgement],
    relevance: LoadedJudgments | None,
) -> Path:
    grounded = [
        judgement
        for key, judgement in judgments.items()
        if items[key].expected_behavior is ExpectedBehavior.GROUNDED
        and judgement.key_fact_recall is not None
    ]
    worst = sorted(
        grounded,
        key=lambda j: (
            -(j.harm == "severe"),
            j.key_fact_recall or 0.0,
            j.faithfulness if j.faithfulness is not None else 1.0,
            j.item_id,
        ),
    )[:ERROR_ROWS]
    lines = [
        "# Worst answers of the full agent",
        "",
        "| item | category | nugget recall | harm | faithfulness | error at | question |",
        "|---|---|---|---|---|---|---|",
    ]
    for judgement in worst:
        item = items[judgement.item_id]
        faithfulness = (
            "--" if judgement.faithfulness is None else f"{judgement.faithfulness:.2f}"
        )
        stage = error_label(
            item, answers[judgement.item_id], _accepted(item, relevance)
        )
        question = item.question.replace("|", "/").replace("\n", " ")
        lines.append(
            f"| {item.item_id} | {item.category.value} "
            f"| {judgement.key_fact_recall:.2f} | {judgement.harm or '--'} "
            f"| {faithfulness} | {stage} | {question} |"
        )
    path = directory / "errors.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _provenance(
    directory: Path,
    run_root: Path,
    configs: Sequence[str],
    relevance: LoadedJudgments | None,
) -> Path:
    runs: dict[str, dict[str, object]] = {}
    for config in configs:
        folder = config_dir(run_root, config)
        runs[config] = {
            name: json.loads((folder / file).read_text(encoding="utf-8"))
            for name, file in (("run", RUN_FILE), ("judge", JUDGE_FILE))
            if (folder / file).is_file()
        }
    path = directory / "provenance.json"
    path.write_text(
        json.dumps(
            {
                "configs": runs,
                "relevance_judgments_sha256": None
                if relevance is None
                else relevance.sha256,
                "bootstrap_resamples": RESAMPLES,
                "seed": SEED,
                "intervals": "Wilson for rates, percentile bootstrap for means",
                "tests": "paired sign-flip randomization, Holm across ablations",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def write_report(
    run_root: Path,
    items: dict[str, GoldenItem],
    relevance: LoadedJudgments | None = None,
) -> list[Path]:
    """Write every table; `relevance` extends relevant chunks as in retrieval metrics."""
    loaded = _load(run_root)
    directory = Path(run_root) / REPORTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    values = {
        config: _values(items, answers, judgments, relevance)
        for config, (answers, judgments) in loaded.items()
    }
    configs = list(loaded)
    written = _main_tables(directory, configs, values)
    if E2EConfig.FULL.value in values:
        written += _ablation_tables(directory, configs, values)
    written.append(_group_table(directory, configs, items, loaded, values))
    written += _calibration_tables(directory, run_root)
    if E2EConfig.FULL.value in loaded:
        answers, judgments = loaded[E2EConfig.FULL.value]
        written.append(_error_analysis(directory, items, answers, judgments, relevance))
    written.append(_provenance(directory, run_root, configs, relevance))
    return written
