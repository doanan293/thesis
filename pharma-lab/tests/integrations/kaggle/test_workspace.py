from pathlib import Path

from pharma_lab.integrations.kaggle.workspace import managed_staging_directory


def test_managed_staging_directory_is_removed_on_exit(tmp_path):
    with managed_staging_directory(tmp_path, prefix="stage-") as staging:
        marker = Path(staging) / "marker"
        marker.write_text("ok", encoding="utf-8")
        assert marker.is_file()
    assert not staging.exists()
