import json
from pathlib import Path

import pytest

from pharma_lab.e2e.golden import GoldenItem, archive_path
from pharma_lab.e2e.golden_changes import SUPERSEDED, GoldenChange, reconcile_golden
from pharma_lab.e2e.harness import run_items
from pharma_lab.e2e.records import (
    ANSWERS_FILE,
    JUDGMENTS_FILE,
    AnswerRecord,
    JsonlStore,
    Judgement,
)
from pharma_lab.e2e.run_identity import RUN_FILE
from pharma_lab.evaluation.artifact_contracts import sha256_file

SECTION = "drug:paracetamol:lieu-dung"


def item(item_id: str, question: str, fact: str = "Tối đa 4 g/ngày") -> GoldenItem:
    return GoldenItem.model_validate(
        {
            "item_id": item_id,
            "category": "answerable",
            "turns": [{"role": "user", "text": question}],
            "expected_behavior": "grounded",
            "gold_section_ids": [SECTION],
            "reference": {
                "key_facts": [
                    {"fact": fact, "evidence_quote": "4 g", "section_id": SECTION}
                ]
            },
        }
    )


def write_golden(path: Path, items: list[GoldenItem]) -> None:
    path.write_text(
        "".join(
            json.dumps(entry.model_dump(mode="json"), ensure_ascii=False) + "\n"
            for entry in items
        ),
        encoding="utf-8",
    )


def make_run(directory: Path, golden: Path, ids: list[str]) -> None:
    directory.mkdir(parents=True)
    (directory / RUN_FILE).write_text(
        json.dumps({"golden_sha256": sha256_file(golden), "config": "full"}),
        encoding="utf-8",
    )
    answers = JsonlStore(directory / ANSWERS_FILE, AnswerRecord)
    judgments = JsonlStore(directory / JUDGMENTS_FILE, Judgement)
    for key in ids:
        answers.append(AnswerRecord(item_id=key, config="full", status="completed"))
        judgments.append(Judgement(item_id=key, config="full", category="answerable"))


def revise(golden: Path, items: list[GoldenItem]) -> None:
    """What `golden build` does: archive the old version, then replace it."""
    golden.replace(archive_path(golden, sha256_file(golden)))
    write_golden(golden, items)


def test_only_changed_items_are_answered_or_judged_again(tmp_path: Path) -> None:
    golden = tmp_path / "golden_e2e.jsonl"
    same = item("e2e-ans-0001", "Liều paracetamol?")
    old_question = item("e2e-ans-0002", "ten biet duoc nay la hoat chat gi?")
    old_reference = item("e2e-ans-0003", "Liều tối đa?", fact="Sai phạm vi")
    write_golden(golden, [same, old_question, old_reference])
    run = tmp_path / "run" / "full"
    make_run(run, golden, ["e2e-ans-0001", "e2e-ans-0002", "e2e-ans-0003"])
    revised = [
        same,
        item("e2e-ans-0002", "Conpac là biệt dược của hoạt chất nào?"),
        item("e2e-ans-0003", "Liều tối đa?"),
    ]
    revise(golden, revised)

    change = reconcile_golden(run, golden, revised)

    assert (change.reanswer, change.rejudge) == (1, 2)
    answers = JsonlStore(run / ANSWERS_FILE, AnswerRecord).latest()
    judgments = JsonlStore(run / JUDGMENTS_FILE, Judgement).latest()
    assert [key for key, a in answers.items() if a.status == SUPERSEDED] == [
        "e2e-ans-0002"
    ]
    assert sorted(key for key, j in judgments.items() if j.error) == [
        "e2e-ans-0002",
        "e2e-ans-0003",
    ]
    stored = json.loads((run / RUN_FILE).read_text("utf-8"))
    assert stored["golden_sha256"] == sha256_file(golden)
    # Reconciling again is a no-op once the run has moved to the new version.
    assert reconcile_golden(run, golden, revised) == GoldenChange(0, 0)


def test_an_unarchived_golden_version_cannot_be_resumed(tmp_path: Path) -> None:
    golden = tmp_path / "golden_e2e.jsonl"
    write_golden(golden, [item("e2e-ans-0001", "Liều paracetamol?")])
    run = tmp_path / "run" / "full"
    make_run(run, golden, ["e2e-ans-0001"])
    write_golden(golden, [item("e2e-ans-0001", "Liều dùng paracetamol?")])

    with pytest.raises(ValueError, match="not archived"):
        reconcile_golden(run, golden, [item("e2e-ans-0001", "Liều dùng?")])


class EchoExecutor:
    def __init__(self) -> None:
        self.asked: list[str] = []

    async def execute(self, item: GoldenItem) -> AnswerRecord:
        self.asked.append(item.item_id)
        return AnswerRecord(item_id=item.item_id, config="full", status="completed")


async def test_superseded_answers_run_again_without_retry_errors(
    tmp_path: Path,
) -> None:
    store = JsonlStore(tmp_path / ANSWERS_FILE, AnswerRecord)
    store.append(
        AnswerRecord(item_id="e2e-ans-0001", config="full", status="completed")
    )
    store.append(
        AnswerRecord(
            item_id="e2e-ans-0002", config="full", status=SUPERSEDED, retryable=True
        )
    )
    executor = EchoExecutor()

    summary = await run_items(
        [item("e2e-ans-0001", "a?"), item("e2e-ans-0002", "b?")],
        executor,
        store,
        config="full",
        concurrency=2,
        retry_errors=False,
    )

    assert executor.asked == ["e2e-ans-0002"]
    assert (summary.ran, summary.errors) == (1, 0)
