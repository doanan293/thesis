from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path

    @classmethod
    def create(cls, work_root: Path, *, build_id: str) -> ArtifactPaths:
        root = Path(work_root) / f"build-{build_id}"
        if root.exists():
            raise FileExistsError(f"Build workspace already exists: {root}")
        root.mkdir(parents=True)
        paths = cls(root=root)
        for directory in paths.directories():
            directory.mkdir(parents=True, exist_ok=True)
        return paths

    @property
    def raw_text(self) -> Path:
        return self.root / "text" / "full.md"

    @property
    def cleaned_text(self) -> Path:
        return self.root / "text" / "full.cleaned.md"

    @property
    def rag_dir(self) -> Path:
        return self.root / "rag"

    @property
    def ankhang_html_dir(self) -> Path:
        return self.root / "ankhang-html"

    @property
    def ankhang_markdown_dir(self) -> Path:
        return self.root / "ankhang-markdown"

    @property
    def canonical_dir(self) -> Path:
        return self.root / "canonical"

    @property
    def source_final_dir(self) -> Path:
        return self.root / "source-final"

    @property
    def candidate_final_dir(self) -> Path:
        return self.root / "candidate-final"

    @property
    def state_path(self) -> Path:
        return self.root / "build-state.json"

    def directories(self) -> tuple[Path, ...]:
        return (
            self.raw_text.parent,
            self.rag_dir,
            self.ankhang_html_dir,
            self.ankhang_markdown_dir,
            self.canonical_dir,
            self.source_final_dir,
            self.candidate_final_dir,
        )

    def cleanup(self) -> None:
        shutil.rmtree(self.root)


def retain_failed_workspace(paths: ArtifactPaths) -> Path:
    failed_root = paths.root.parent / "failed"
    latest = failed_root / "latest"
    if latest.exists():
        shutil.rmtree(latest)
    failed_root.mkdir(parents=True, exist_ok=True)
    paths.root.replace(latest)
    return latest
