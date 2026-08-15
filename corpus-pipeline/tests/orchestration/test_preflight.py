from corpus_pipeline.config.enums import Backend
from corpus_pipeline.orchestration.preflight import run_preflight


def test_missing_heavy_pdf_requests_archive_restore(tmp_path):
    report = run_preflight(
        Backend.LOCAL,
        project_root=tmp_path,
        env_file=tmp_path / ".env",
    )

    issue = next(item for item in report.issues if item.check == "raw-inputs")
    assert "data/heavy/raw/duoc-thu-quoc-gia-viet-nam.pdf" in (issue.resource or "")
    assert "restore data/heavy" in issue.message.lower()
