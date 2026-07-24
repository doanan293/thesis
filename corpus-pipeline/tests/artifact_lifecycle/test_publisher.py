from pathlib import Path

import pytest

from artifact_lifecycle.publisher import publish_contract, recover_publish


def _contract(path: Path, marker: str) -> Path:
    path.mkdir()
    for name in (
        "sections.jsonl",
        "chunks.jsonl",
        "manifest.json",
        "validation_report.json",
    ):
        (path / name).write_text(marker, encoding="utf-8")
    return path


def test_publish_replaces_complete_directory(tmp_path: Path) -> None:
    final = _contract(tmp_path / "rag-final", "old")
    candidate = _contract(tmp_path / "candidate", "new")

    publish_contract(candidate, final, validate=lambda _: None)

    assert (final / "chunks.jsonl").read_text() == "new"
    assert not (tmp_path / ".rag-final.previous").exists()


def test_publish_rolls_back_when_activation_fails(tmp_path: Path) -> None:
    final = _contract(tmp_path / "rag-final", "old")
    candidate = _contract(tmp_path / "candidate", "new")

    def fail(stage: str) -> None:
        if stage == "after_backup":
            raise RuntimeError("injected")

    with pytest.raises(RuntimeError, match="injected"):
        publish_contract(candidate, final, validate=lambda _: None, failpoint=fail)

    assert (final / "chunks.jsonl").read_text() == "old"


def test_recover_restores_previous_when_final_is_missing(tmp_path: Path) -> None:
    _contract(tmp_path / ".rag-final.previous", "old")
    final = tmp_path / "rag-final"
    recover_publish(final)
    assert final.exists()
    assert not (tmp_path / ".rag-final.previous").exists()
