from __future__ import annotations

import json
import zipfile
from pathlib import Path

from seed_pipeline.integrations.kaggle.dataset_service import (
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


def test_version_dataset_uses_the_given_message(tmp_path, monkeypatch):
    runner = RecordingRunner()
    service = DatasetService(runner, "owner")
    monkeypatch.setattr(
        service,
        "inspect_state",
        lambda *_args, **_kwargs: DatasetRemoteState(
            DatasetPresence.EXISTS, current_version=3
        ),
    )

    service.ensure_dataset("slug", "Title", tmp_path, message="Refresh data")

    command = runner.calls[-1][0]
    assert command[command.index("-m") + 1] == "Refresh data"


class ZippingRunner(RecordingRunner):
    """Serves each requested file the way the Kaggle API serves large files."""

    def run(self, args, capture_output=False, *, live_output=False):
        super().run(args, capture_output, live_output=live_output)
        name = args[args.index("-f") + 1]
        destination = Path(args[args.index("-p") + 1])
        with zipfile.ZipFile(destination / f"{name}.zip", "w") as archive:
            archive.writestr(name, b"payload")
        return ""


def test_download_file_unpacks_a_zipped_single_file(tmp_path):
    service = DatasetService(ZippingRunner(), "owner")

    path = service.download_file("owner/slug", "data.part-0001", tmp_path)

    assert path == tmp_path / "data.part-0001"
    assert path.read_bytes() == b"payload"
    assert not (tmp_path / "data.part-0001.zip").exists()
