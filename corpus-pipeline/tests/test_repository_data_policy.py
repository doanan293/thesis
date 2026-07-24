import subprocess
from pathlib import Path

from config.paths import (
    RAG_FINAL_CHUNKS_PATH,
    RAG_FINAL_SECTIONS_PATH,
    RAW_ANKHANG_SNAPSHOTS_DIR,
    RAW_DIR,
)


def _attribute(path: Path, name: str) -> str:
    repository_root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    relative = Path(path).resolve().relative_to(repository_root)
    result = subprocess.run(
        ["git", "check-attr", name, "--", str(relative)],
        check=True,
        capture_output=True,
        text=True,
        cwd=repository_root,
    )
    return result.stdout.strip().rsplit(": ", 1)[-1]


def test_large_durable_artifacts_use_lfs() -> None:
    assert _attribute(RAW_DIR / "duoc-thu-quoc-gia-viet-nam.pdf", "filter") == "lfs"
    assert (
        _attribute(RAW_ANKHANG_SNAPSHOTS_DIR / "ankhang-test.tar.zst", "filter")
        == "lfs"
    )
    assert _attribute(RAG_FINAL_SECTIONS_PATH, "filter") == "lfs"
    assert _attribute(RAG_FINAL_CHUNKS_PATH, "filter") == "lfs"


def test_workspace_and_unpacked_html_are_ignored() -> None:
    for path in (
        Path("data/.work/build-test/state.json"),
        Path("data/raw/ankhang/html/test.html"),
        Path("data/cache/vector_embeddings/test.jsonl"),
        Path("data/runs/test/result.json"),
    ):
        result = subprocess.run(
            ["git", "check-ignore", "-q", str(path)],
            check=False,
        )
        assert result.returncode == 0, path
