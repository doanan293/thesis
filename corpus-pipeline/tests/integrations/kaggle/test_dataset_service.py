from __future__ import annotations

import json

from corpus_pipeline.integrations.kaggle.dataset_service import (
    DatasetPresence,
    DatasetRemoteState,
    DatasetService,
)


class RecordingRunner:
    def __init__(self, status_output: str = ""):
        self.calls = []
        self.status_output = status_output

    def run(self, args, capture_output=False, *, live_output=False):
        self.calls.append((args, capture_output, live_output))
        if "status" in args:
            return self.status_output
        return ""


def test_create_dataset_streams_progress_and_lifecycle(tmp_path, monkeypatch, capsys):
    runner = RecordingRunner()
    service = DatasetService(runner, "owner")
    monkeypatch.setattr(
        service,
        "inspect_state",
        lambda *_args, **_kwargs: DatasetRemoteState(DatasetPresence.ABSENT),
    )

    result = service.ensure_dataset("slug", "Title", tmp_path)

    assert result.expected_version == 1
    assert runner.calls[-1][2] is True
    assert runner.calls[-1][0][1:3] == ["datasets", "create"]
    assert capsys.readouterr().out == (
        "Uploading Kaggle dataset owner/slug (create)...\n"
        "Uploaded Kaggle dataset owner/slug.\n"
    )


def test_version_dataset_streams_progress_and_lifecycle(tmp_path, monkeypatch, capsys):
    runner = RecordingRunner()
    service = DatasetService(runner, "owner")
    monkeypatch.setattr(
        service,
        "inspect_state",
        lambda *_args, **_kwargs: DatasetRemoteState(
            DatasetPresence.EXISTS, current_version=3
        ),
    )

    result = service.ensure_dataset("slug", "Title", tmp_path)

    assert result.expected_version == 4
    assert runner.calls[-1][2] is True
    assert runner.calls[-1][0][1:3] == ["datasets", "version"]
    assert capsys.readouterr().out == (
        "Uploading Kaggle dataset owner/slug (version)...\n"
        "Uploaded Kaggle dataset owner/slug.\n"
    )


def test_inspect_state_keeps_status_command_silent(tmp_path, capsys):
    del tmp_path
    runner = RecordingRunner(
        json.dumps({"status": "READY", "current_version_number": 2})
    )
    service = DatasetService(runner, "owner")

    state = service.inspect_state("owner/slug")

    assert state.presence is DatasetPresence.EXISTS
    assert state.status == "READY"
    assert state.current_version == 2
    assert runner.calls[-1][2] is False
    assert capsys.readouterr().out == ""
