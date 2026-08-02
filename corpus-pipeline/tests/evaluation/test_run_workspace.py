from dataclasses import replace

import pytest

from corpus_pipeline.evaluation.run_workspace import (
    RunConflictError,
    RunIdentity,
    RunWorkspace,
    load_run_record,
)


def _identity():
    return RunIdentity(
        "evaluation.jsonl",
        "eval",
        "collection",
        "model",
        "queries",
        "hybrid",
        50,
        60,
        50,
    )


def test_workspace_reuses_equal_identity_and_rejects_conflict(tmp_path):
    identity = _identity()
    workspace = RunWorkspace.open_or_create(tmp_path / "smoke", identity)
    assert (
        RunWorkspace.open_or_create(tmp_path / "smoke", identity).root == workspace.root
    )
    with pytest.raises(RunConflictError, match="another --run name"):
        RunWorkspace.open_or_create(tmp_path / "smoke", replace(identity, limit=None))


def test_reopening_equal_identity_preserves_stage_metadata(tmp_path):
    identity = _identity()
    workspace = RunWorkspace.open_or_create(tmp_path / "smoke", identity)
    artifact = type(
        "Artifact", (), {"data_path": workspace.candidates_dir / "candidates.jsonl"}
    )()
    workspace.record_candidates(artifact)

    RunWorkspace.open_or_create(workspace.root, identity)

    record = load_run_record(workspace.root / "run.json")
    assert record.status == "retrieved"
    assert record.candidates_dir == str(workspace.candidates_dir)
