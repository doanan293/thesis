import json
import math
from pathlib import Path

import pytest

from pharma_lab.e2e.calibration import (
    Grade,
    bootstrap_interval,
    calibration_dir,
    cohen_kappa,
    export_calibration,
    mean_abs_diff,
    score_calibration,
    spearman,
)
from pharma_lab.e2e.golden import GoldenItem
from pharma_lab.e2e.records import AnswerRecord, JsonlStore, Judgement

SECTION = "drug:x:lieu-dung"


def golden(item_id: str, category: str) -> GoldenItem:
    grounded = category in ("answerable", "multi_turn")
    turns = [{"role": "user", "text": f"hỏi {item_id}"}]
    if category == "multi_turn":
        turns = [
            {"role": "user", "text": "a"},
            {"role": "assistant", "text": "b"},
            {"role": "user", "text": "c"},
        ]
    behaviour = {
        "answerable": "grounded",
        "multi_turn": "grounded",
        "unanswerable": "abstain",
        "out_of_scope": "redirect",
        "injection": "blocked",
    }[category]
    data: dict = {
        "item_id": item_id,
        "category": category,
        "turns": turns,
        "expected_behavior": behaviour,
    }
    if grounded:
        data["gold_section_ids"] = [SECTION]
        data["reference"] = {
            "answer": "r",
            "key_facts": [
                {"fact": "f1", "evidence_quote": "q", "section_id": SECTION},
                {"fact": "f2", "evidence_quote": "q", "section_id": SECTION},
            ],
        }
    if category == "unanswerable":
        data["absent_terms"] = ["zz"]
    return GoldenItem.model_validate(data)


def golden_set() -> dict[str, GoldenItem]:
    items = [golden(f"e2e-ans-{n:04d}", "answerable") for n in range(40)]
    items += [golden(f"e2e-inj-{n:04d}", "injection") for n in range(20)]
    return {item.item_id: item for item in items}


def write_run(root: Path, items: dict[str, GoldenItem]) -> None:
    for config in ("full", "one-step"):
        store = JsonlStore(root / config / "answers.jsonl", AnswerRecord)
        for key in items:
            store.append(
                AnswerRecord(
                    item_id=key,
                    config=config,
                    status="completed",
                    answer_mode="grounded" if key.startswith("e2e-ans") else "blocked",
                    answer_text="Có [1].",
                    context_text="[1] A\nnội dung",
                )
            )


def test_export_is_a_blind_random_sample_per_config(tmp_path: Path) -> None:
    items = golden_set()
    write_run(tmp_path, items)

    path = export_calibration(tmp_path, items, seed=7)
    first = path.read_text("utf-8")
    export_calibration(tmp_path, items, seed=7)

    assert path.read_text("utf-8") == first
    rows = [json.loads(line) for line in first.splitlines()]
    assert len(rows) == 100
    assert "config" not in rows[0] and "faithfulness" not in rows[0]
    assert rows[0]["cited_sentences"] == ["Có [1]."]
    key = json.loads((calibration_dir(tmp_path) / "key.json").read_text("utf-8"))
    configs = [entry["config"] for entry in key.values()]
    assert configs.count("full") == configs.count("one-step") == 50
    # A simple random sample (for PPI), not a stratified one.
    full_items = {e["item_id"] for e in key.values() if e["config"] == "full"}
    assert len(full_items) == 50
    assert export_calibration(tmp_path, items, seed=8).read_text("utf-8") != first


def test_export_needs_enough_answers(tmp_path: Path) -> None:
    items = dict(list(golden_set().items())[:10])
    write_run(tmp_path, items)
    with pytest.raises(ValueError, match="need 50 full answers"):
        export_calibration(tmp_path, items)


def test_kappa_and_spearman_known_values() -> None:
    assert cohen_kappa([True, False, True], [True, False, True]) == 1.0
    # observed 0.5, expected 0.5 -> 0
    assert cohen_kappa([True, True, False, False], [True, False, True, False]) == 0.0
    assert cohen_kappa([True, True], [True, True]) == 1.0
    assert spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert math.isnan(spearman([1, 1, 1], [1, 2, 3]))


def test_mean_absolute_difference_known_value() -> None:
    assert mean_abs_diff([1.0, 0.5, 0.0], [0.8, 0.5, 0.3]) == pytest.approx(
        (0.2 + 0.0 + 0.3) / 3
    )


def test_bootstrap_interval_brackets_a_perfect_statistic() -> None:
    pairs = [(float(n), float(n)) for n in range(20)]
    low, high = bootstrap_interval(
        pairs,
        lambda a, b: spearman([float(str(x)) for x in a], [float(str(y)) for y in b]),
    )
    assert low == pytest.approx(1.0) and high == pytest.approx(1.0)


def test_score_compares_grades_with_judgments(tmp_path: Path) -> None:
    directory = calibration_dir(tmp_path)
    directory.mkdir(parents=True)
    key = {}
    grades = []
    judgments = JsonlStore(tmp_path / "full" / "judgments.jsonl", Judgement)
    for n in range(10):
        blind = f"cal-{n:03d}"
        item = f"e2e-ans-{n:04d}"
        key[blind] = {"config": "full", "item_id": item}
        verdicts = ["supported", "missing"] if n % 2 else ["supported", "supported"]
        judgments.append(
            Judgement(
                item_id=item,
                config="full",
                category="answerable",
                key_fact_verdicts=verdicts,
                contradiction=False,
                key_fact_recall=verdicts.count("supported") / 2,
                faithfulness=n / 10,
                citation_support=1.0 if n % 3 else 0.0,
            )
        )
        grades.append(
            Grade(
                blind_id=blind,
                grader="claude-opus-5",
                key_fact_verdicts=verdicts,
                faithfulness=n / 10,
                citation_checks={1: bool(n % 3)},
            )
        )
    (directory / "key.json").write_text(json.dumps(key), encoding="utf-8")
    (directory / "grades.jsonl").write_text(
        "".join(g.model_dump_json() + "\n" for g in grades), encoding="utf-8"
    )

    rows = {(row.metric, row.statistic): row for row in score_calibration(tmp_path)}

    kappa = rows[("key_fact_supported", "cohen_kappa")]
    assert (kappa.value, kappa.n, kappa.reliable) == (1.0, 20, True)
    assert rows[("key_fact_supported", "percent_agreement")].value == 1.0
    assert rows[("key_fact_supported", "gwet_ac1")].value == 1.0
    assert rows[("faithfulness", "spearman")].value == pytest.approx(1.0)
    assert rows[("supported_citation_rate", "spearman")].value == pytest.approx(1.0)
    # Bounded scores near the ceiling: the level of agreement is reported too.
    mad = rows[("faithfulness", "mean_abs_diff")]
    assert mad.value == pytest.approx(0.0) and not mad.reliable
    assert (
        rows[("nugget_recall", "mean_abs_diff")].n
        == rows[("nugget_recall", "spearman")].n
    )
    assert rows[("injection_followed", "cohen_kappa")].n == 0
    assert not rows[("injection_followed", "cohen_kappa")].reliable
    stored = json.loads((directory / "agreement.json").read_text("utf-8"))
    assert any(row["value"] is None for row in stored["rows"])
    ppi = json.loads((directory / "ppi.json").read_text("utf-8"))["rows"]
    recall = next(row for row in ppi if row["metric"] == "nugget_recall")
    # Identical judge and grader labels: the correction is zero.
    assert recall["ppi"] == pytest.approx(recall["judge_mean"])
    assert recall["n_labelled"] == 10
