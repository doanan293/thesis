import subprocess
from pathlib import Path

from seed_pipeline.config.paths import PROJECT_ROOT


def _is_ignored(path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", path],
        cwd=PROJECT_ROOT,
        check=False,
    )
    return result.returncode == 0


def test_data_and_local_tool_ignore_boundaries():
    assert _is_ignored("data/heavy/probe.bin")
    assert not _is_ignored("data/runtime_kaggle_profiles/probe.json")
    assert not _is_ignored("data/retrieval_eval/probe/run.json")
    # Specs and plans are committed, as in the other projects.
    assert not _is_ignored("docs/superpowers/specs/probe.md")


def test_no_tracked_data_file_exceeds_five_mib():
    output = subprocess.run(
        ["git", "ls-files", "-z", "--", "data"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    tracked = [Path(item.decode()) for item in output.split(b"\0") if item]
    oversized = [
        path
        for path in tracked
        if (PROJECT_ROOT / path).stat().st_size > 5 * 1024 * 1024
    ]
    assert oversized == []
