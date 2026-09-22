"""Carry a run over to a revised golden set, redoing only what the revision touched.

A run records the SHA-256 of the golden set it was made with. When the golden set
is rebuilt, `golden build` archives the previous version, so a later `e2e run` or
`e2e judge` can compare the two versions item by item:

- an item whose turns changed gets its answer and judgement superseded, so it is
  answered and judged again;
- an item whose reference or expected behaviour changed gets only its judgement
  superseded, so it is judged again from the stored answer;
- every other item keeps its answer and judgement.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from pharma_lab.e2e.golden import GoldenItem, archive_path, load_golden
from pharma_lab.e2e.records import (
    ANSWERS_FILE,
    JUDGMENTS_FILE,
    AnswerRecord,
    JsonlStore,
    Judgement,
)
from pharma_lab.e2e.run_identity import RUN_FILE
from pharma_lab.evaluation.artifact_contracts import sha256_file, write_json

SUPERSEDED = "superseded"
SUPERSEDED_ERROR = "golden item changed"


def _digest(payload: object) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def answer_inputs(item: GoldenItem) -> str:
    """Fingerprint of everything the pipeline sees: the conversation."""
    return _digest([turn.model_dump(mode="json") for turn in item.turns])


def judge_inputs(item: GoldenItem) -> str:
    """Fingerprint of everything the judge compares an answer against."""
    return _digest(
        item.model_dump(
            mode="json",
            include={"category", "expected_behavior", "turns", "reference"},
        )
    )


@dataclass(frozen=True)
class GoldenChange:
    reanswer: int
    rejudge: int


def reconcile_golden(
    directory: Path, golden_path: Path, items: list[GoldenItem]
) -> GoldenChange:
    """Supersede the records of changed items and move the run to `golden_path`."""
    run_file = Path(directory) / RUN_FILE
    if not run_file.is_file():
        return GoldenChange(reanswer=0, rejudge=0)
    stored = json.loads(run_file.read_text(encoding="utf-8"))
    recorded = stored["golden_sha256"]
    current = sha256_file(golden_path)
    if recorded == current:
        return GoldenChange(reanswer=0, rejudge=0)
    previous_path = archive_path(golden_path, recorded)
    if not previous_path.is_file():
        raise ValueError(
            f"{run_file} was made with golden set {recorded[:12]}, which is not "
            f"archived at {previous_path}; start a new --run"
        )
    previous = {item.item_id: item for item in load_golden(previous_path)}
    answers = JsonlStore(Path(directory) / ANSWERS_FILE, AnswerRecord)
    judgments = JsonlStore(Path(directory) / JUDGMENTS_FILE, Judgement)
    answered = answers.latest()
    judged = judgments.latest()
    config = stored["config"]
    reanswer = rejudge = 0
    for item in items:
        before = previous.get(item.item_id)
        turns_changed = before is None or answer_inputs(before) != answer_inputs(item)
        if turns_changed and item.item_id in answered:
            answers.append(
                AnswerRecord(
                    item_id=item.item_id,
                    config=config,
                    status=SUPERSEDED,
                    error=SUPERSEDED_ERROR,
                    retryable=True,
                )
            )
            reanswer += 1
        reference_changed = before is None or judge_inputs(before) != judge_inputs(item)
        if (turns_changed or reference_changed) and item.item_id in judged:
            judgments.append(
                Judgement(
                    item_id=item.item_id,
                    config=config,
                    category=str(item.category),
                    error=SUPERSEDED_ERROR,
                )
            )
            rejudge += 1
    write_json(run_file, {**stored, "golden_sha256": current})
    return GoldenChange(reanswer=reanswer, rejudge=rejudge)
