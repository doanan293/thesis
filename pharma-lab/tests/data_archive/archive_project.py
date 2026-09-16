"""A throwaway Git project whose data/ mixes tracked, ignored and work files."""

import hashlib
import subprocess
from pathlib import Path


def noise(seed: str, size: int) -> bytes:
    """Deterministic bytes that zstd cannot shrink, so small parts still split."""
    blocks = (
        hashlib.sha256(f"{seed}:{index}".encode()).digest()
        for index in range(size // 32 + 1)
    )
    return b"".join(blocks)[:size]


FILES = {
    "cache/nested/scores.bin": noise("scores", 3000),
    "cache/text_embeddings/model.jsonl": noise("embeddings", 6000),
}


def git(root: Path, *args: str) -> None:
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            *args,
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )


def make_project(root: Path, *, with_data: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    git(root, "init", "-q")
    (root / ".gitignore").write_text("data/cache/\ndata/work/\n", encoding="utf-8")
    (root / "data").mkdir()
    (root / "data" / "README.md").write_text("tracked\n", encoding="utf-8")
    git(root, "add", ".gitignore", "data/README.md")
    git(root, "commit", "-q", "-m", "init")
    if with_data:
        for relative, payload in FILES.items():
            path = root / "data" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        lock = root / "data" / "work" / "locks" / "probe.job.lock"
        lock.parent.mkdir(parents=True)
        lock.write_text("lock", encoding="utf-8")
    return root
