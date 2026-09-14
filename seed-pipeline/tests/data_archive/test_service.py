import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from tests.data_archive.archive_project import FILES, make_project

from seed_pipeline.data_archive.manifest import ARCHIVE_MANIFEST_NAME
from seed_pipeline.data_archive.service import pull_data, push_data
from seed_pipeline.integrations.kaggle.dataset_service import DatasetService
from seed_pipeline.integrations.kaggle.errors import KaggleCommandError

NOW = datetime(2026, 9, 14, tzinfo=UTC)


class FakeKaggle:
    """A private dataset kept in a local folder, driven by Kaggle CLI arguments."""

    def __init__(self, remote: Path) -> None:
        self.remote = remote
        self.version = 0
        self.calls: list[list[str]] = []

    def run(
        self,
        args: list[str],
        /,
        capture_output: bool = False,
        *,
        live_output: bool = False,
    ) -> str:
        del capture_output, live_output
        self.calls.append(args)
        action = args[1:3]
        if action == ["datasets", "status"]:
            if self.version == 0:
                raise KaggleCommandError(
                    operation="status",
                    target=args[3],
                    returncode=1,
                    stdout="",
                    stderr="404 Not Found",
                )
            return json.dumps(
                {"status": "READY", "current_version_number": self.version}
            )
        if action in (["datasets", "create"], ["datasets", "version"]):
            shutil.rmtree(self.remote, ignore_errors=True)
            shutil.copytree(Path(args[args.index("-p") + 1]), self.remote)
            self.version += 1
            return ""
        if action == ["datasets", "download"]:
            name = args[args.index("-f") + 1]
            destination = Path(args[args.index("-p") + 1])
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.remote / name, destination / name)
            return ""
        raise AssertionError(f"unexpected Kaggle command: {args}")


def test_push_then_pull_restores_the_ignored_files(tmp_path: Path) -> None:
    kaggle = FakeKaggle(tmp_path / "remote")
    dataset = DatasetService(kaggle, "owner")
    source = make_project(tmp_path / "source")

    pushed = push_data(
        project_root=source, dataset=dataset, message="First", now=NOW, part_size=1024
    )

    assert (pushed.reference, pushed.version, pushed.file_count) == (
        "owner/seed-pipeline-data",
        1,
        len(FILES),
    )
    create = next(call for call in kaggle.calls if call[1:3] == ["datasets", "create"])
    assert "--public" not in create
    assert create[create.index("--dir-mode") + 1] == "skip"
    assert (kaggle.remote / ARCHIVE_MANIFEST_NAME).is_file()

    clone = make_project(tmp_path / "clone", with_data=False)
    pulled = pull_data(project_root=clone, dataset=dataset, force=False)

    assert (pulled.written, pulled.unchanged) == (len(FILES), 0)
    assert pulled.downloaded_parts == pushed.part_count
    for relative, payload in FILES.items():
        assert (clone / "data" / relative).read_bytes() == payload

    again = pull_data(project_root=clone, dataset=dataset, force=False)

    assert (again.written, again.unchanged, again.downloaded_parts) == (
        0,
        len(FILES),
        0,
    )


def test_a_second_push_creates_a_version_with_the_message(tmp_path: Path) -> None:
    kaggle = FakeKaggle(tmp_path / "remote")
    dataset = DatasetService(kaggle, "owner")
    source = make_project(tmp_path / "source")
    push_data(project_root=source, dataset=dataset, message="First", now=NOW)

    second = push_data(project_root=source, dataset=dataset, message="Refresh", now=NOW)

    version = kaggle.calls[-1]
    assert version[1:3] == ["datasets", "version"]
    assert version[version.index("-m") + 1] == "Refresh"
    assert second.version == 2
