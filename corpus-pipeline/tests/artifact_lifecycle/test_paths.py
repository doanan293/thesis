from pathlib import Path

from corpus_pipeline.artifacts.paths import ArtifactPaths, retain_failed_workspace


def test_artifact_paths_are_isolated_under_one_workspace(tmp_path: Path) -> None:
    paths = ArtifactPaths.create(tmp_path / ".work", build_id="abc123")

    assert paths.root == tmp_path / ".work" / "build-abc123"
    assert paths.raw_text == paths.root / "text" / "full.md"
    assert paths.cleaned_text == paths.root / "text" / "full.cleaned.md"
    assert paths.rag_dir == paths.root / "rag"
    assert paths.ankhang_html_dir == paths.root / "ankhang-html"
    assert paths.ankhang_markdown_dir == paths.root / "ankhang-markdown"
    assert paths.canonical_dir == paths.root / "canonical"
    assert paths.source_final_dir == paths.root / "source-final"
    assert paths.candidate_final_dir == paths.root / "candidate-final"
    assert all(paths.root in path.parents for path in paths.directories())


def test_retain_failed_workspace_keeps_only_latest(tmp_path: Path) -> None:
    work_root = tmp_path / ".work"
    first = ArtifactPaths.create(work_root, build_id="first")
    (first.root / "marker.txt").write_text("first", encoding="utf-8")
    retain_failed_workspace(first)

    second = ArtifactPaths.create(work_root, build_id="second")
    (second.root / "marker.txt").write_text("second", encoding="utf-8")
    retained = retain_failed_workspace(second)

    assert retained == work_root / "failed" / "latest"
    assert (retained / "marker.txt").read_text(encoding="utf-8") == "second"
    assert not (work_root / "failed" / "previous").exists()
