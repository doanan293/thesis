from pathlib import Path

from seed_pipeline.artifacts.publisher import publish_contract
from seed_pipeline.integrations.kaggle.job_lock import lock_file_name


def _candidate(root: Path) -> Path:
    candidate = root / "candidate"
    candidate.mkdir(parents=True)
    (candidate / "manifest.json").write_text("{}", encoding="utf-8")
    return candidate


def test_publish_keeps_its_lock_under_the_lock_directory(
    tmp_path: Path, isolated_lock_dir: Path
) -> None:
    final = tmp_path / "corpus" / "rag-final"

    publish_contract(_candidate(tmp_path), final, validate=lambda _: None)

    assert (final / "manifest.json").is_file()
    assert [path.name for path in final.parent.iterdir()] == ["rag-final"]
    assert (isolated_lock_dir / lock_file_name(final, "publish")).is_file()
