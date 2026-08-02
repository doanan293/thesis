from pathlib import Path

from corpus_pipeline.cli.options import Backend
from corpus_pipeline.orchestration.preflight import PreflightReport, run_preflight


def test_preflight_reports_all_missing_local_inputs(tmp_path: Path):
    report = run_preflight(Backend.LOCAL, project_root=tmp_path)

    assert isinstance(report, PreflightReport)
    assert report.ok is False
    checks = {issue.check for issue in report.issues}
    assert "raw-inputs" in checks
    assert "compose" in checks
    assert "local-models" in checks


def test_kaggle_preflight_adds_credentials_check(tmp_path: Path):
    report = run_preflight(
        Backend.KAGGLE,
        project_root=tmp_path,
        env_file=tmp_path / "missing.env",
    )

    assert "kaggle-credentials" in {issue.check for issue in report.issues}
