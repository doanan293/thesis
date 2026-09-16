import csv
import json
from pathlib import Path

import pytest

from pharma_lab.e2e.calibration import calibration_dir
from pharma_lab.e2e.golden import GoldenItem
from pharma_lab.e2e.records import (
    AnswerRecord,
    CitedSection,
    JsonlStore,
    Judgement,
    TokenUsage,
)
from pharma_lab.e2e.report import (
    bootstrap_mean,
    error_label,
    paired_delta,
    write_report,
)

SECTION = "drug:x:lieu-dung"


def golden(n: int) -> GoldenItem:
    return GoldenItem.model_validate(
        {
            "item_id": f"e2e-ans-{n:04d}",
            "category": "answerable",
            "turns": [{"role": "user", "text": f"Câu {n} | liều?"}],
            "expected_behavior": "grounded",
            "gold_section_ids": [SECTION],
            "reference": {
                "answer": "r",
                "key_facts": [
                    {"fact": "f", "evidence_quote": "q", "section_id": SECTION}
                ],
            },
            "eval_group": "formulary" if n % 2 else "leaflet",
        }
    )


def answer(n: int, config: str, *, retrieved: str = SECTION) -> AnswerRecord:
    return AnswerRecord(
        item_id=f"e2e-ans-{n:04d}",
        config=config,
        status="completed",
        answer_mode="grounded",
        citations=[
            CitedSection(
                index=1, chunk_version_id="c", chunk_id=None, section_id=SECTION
            )
        ],
        retrieved_chunk_ids=[f"{retrieved}:chunk-001"],
        llm_calls=4 if config == "full" else 2,
        usage_by_role={
            "answer": TokenUsage(calls=1, prompt_tokens=100, completion_tokens=10)
        },
        latency_seconds=float(n),
    )


def judgement(n: int, config: str, recall: float) -> Judgement:
    return Judgement(
        item_id=f"e2e-ans-{n:04d}",
        config=config,
        category="answerable",
        behaviour_correct=True,
        key_fact_recall=recall,
        contradiction=False,
        faithfulness=recall,
        citation_precision=1.0,
        citation_recall=1.0,
    )


def write_config(root: Path, config: str, recalls: list[float]) -> None:
    answers = JsonlStore(root / config / "answers.jsonl", AnswerRecord)
    judgments = JsonlStore(root / config / "judgments.jsonl", Judgement)
    for n, recall in enumerate(recalls):
        answers.append(answer(n, config, retrieved="other" if n == 0 else SECTION))
        judgments.append(judgement(n, config, recall))


def rows(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.read_text("utf-8").splitlines()))


def test_bootstrap_mean_and_paired_delta() -> None:
    summary = bootstrap_mean([1.0, 1.0, 1.0])
    assert (summary.n, summary.mean, summary.ci_low, summary.ci_high) == (
        3,
        1.0,
        1.0,
        1.0,
    )
    empty = bootstrap_mean([])
    assert empty.n == 0
    delta = paired_delta({"a": 1.0, "b": 0.5, "c": 0.0}, {"a": 0.5, "b": 0.0})
    assert delta.n == 2
    assert delta.mean == pytest.approx(-0.5)


def test_error_labels() -> None:
    item = golden(1)
    assert error_label(item, answer(1, "full", retrieved="other")) == "retrieval"
    uncited = answer(1, "full").model_copy(update={"citations": []})
    assert error_label(item, uncited) == "citation"
    assert error_label(item, answer(1, "full")) == "answer"


def test_write_report_builds_every_table(tmp_path: Path) -> None:
    items = {golden(n).item_id: golden(n) for n in range(6)}
    write_config(tmp_path, "full", [1.0, 1.0, 0.5, 1.0, 0.0, 1.0])
    write_config(tmp_path, "one-step", [0.5, 0.5, 0.5, 0.5, 0.0, 0.5])
    directory = calibration_dir(tmp_path)
    directory.mkdir()
    (directory / "agreement.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "metric": "key_fact_supported",
                        "statistic": "cohen_kappa",
                        "n": 20,
                        "value": 0.8,
                        "ci_low": 0.6,
                        "ci_high": 0.9,
                        "reliable": True,
                    },
                    {
                        "metric": "injection_followed",
                        "statistic": "cohen_kappa",
                        "n": 0,
                        "value": None,
                        "ci_low": None,
                        "ci_high": None,
                        "reliable": False,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    written = write_report(tmp_path, items)

    assert sorted(path.name for path in written) == [
        "ablation.csv",
        "ablation.tex",
        "by_group.csv",
        "calibration.csv",
        "calibration.tex",
        "errors.md",
        "main.csv",
        "main.tex",
    ]
    reports = tmp_path / "reports"
    main = rows(reports / "main.csv")
    recall = {r["config"]: r for r in main if r["metric"] == "key_fact_recall"}
    assert float(recall["full"]["mean"]) == pytest.approx(0.75)
    assert float(recall["one-step"]["mean"]) == pytest.approx(5 / 12)
    calls = {r["config"]: float(r["mean"]) for r in main if r["metric"] == "llm_calls"}
    assert calls == {"full": 4.0, "one-step": 2.0}
    ablation = rows(reports / "ablation.csv")
    delta = next(r for r in ablation if r["metric"] == "key_fact_recall")
    assert delta["config"] == "one-step"
    assert float(delta["delta"]) == pytest.approx(5 / 12 - 0.75)
    groups = rows(reports / "by_group.csv")
    assert {r["group"] for r in groups} == {"answerable", "formulary", "leaflet"}
    tex = (reports / "main.tex").read_text("utf-8")
    assert r"key\_fact\_recall" in tex and "latency\\_p95" in tex
    assert "--" in (reports / "calibration.tex").read_text("utf-8")
    errors = (reports / "errors.md").read_text("utf-8").splitlines()
    assert errors[4].startswith("| e2e-ans-0004 | answerable | 0.00 |")
    assert "| retrieval |" in next(line for line in errors if "e2e-ans-0000" in line)
    assert "Câu 4 / liều?" in errors[4]


def test_report_needs_judged_answers(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no judged configuration"):
        write_report(tmp_path, {})
