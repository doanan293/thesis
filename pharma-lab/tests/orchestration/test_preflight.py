from pharma_lab.config.enums import Backend
from pharma_lab.orchestration.preflight import run_preflight


def test_missing_formulary_pdf_requests_archive_restore(tmp_path):
    report = run_preflight(
        Backend.LOCAL,
        project_root=tmp_path,
        env_file=tmp_path / ".env",
    )

    issue = next(item for item in report.issues if item.check == "raw-inputs")
    assert "data/sources/duoc-thu-quoc-gia-viet-nam.pdf" in (issue.resource or "")
    assert "restore data/sources" in issue.message.lower()


def _profile_env(tmp_path, *, include_token=True):
    env_file = tmp_path / ".env"
    token = "KAGGLE_ACC2_API_TOKEN=secondary-token\n" if include_token else ""
    env_file.write_text(
        "KAGGLE_ACCOUNT_DEFAULT=acc1\n"
        "KAGGLE_SHARED_OWNER=primary-user\n"
        "KAGGLE_ACC2_USERNAME=secondary-user\n"
        f"{token}",
        encoding="utf-8",
    )
    return env_file


class FakeRunner:
    def __init__(self, environment):
        self.environment = environment

    def run(self, args, capture_output=False):
        del capture_output
        if args[1:3] == ["config", "view"]:
            return "username: secondary-user\n"
        if args[1:3] == ["datasets", "files"]:
            return "name,size,creationDate\nruntime_manifest.json,100,2026-08-23\n"
        raise AssertionError(args)


def test_profile_preflight_checks_selected_account_and_shared_runtime(tmp_path):
    from pharma_lab.config.enums import Backend

    report = run_preflight(
        Backend.KAGGLE,
        project_root=tmp_path,
        env_file=_profile_env(tmp_path),
        kaggle_account="acc2",
        runner_factory=FakeRunner,
    )

    assert not [issue for issue in report.issues if issue.check == "kaggle-credentials"]


def test_profile_preflight_names_missing_token_without_exposing_it(
    tmp_path, monkeypatch
):
    from pharma_lab.config.enums import Backend

    monkeypatch.delenv("KAGGLE_ACC2_API_TOKEN", raising=False)
    monkeypatch.delenv("KAGGLE_ACC2_USERNAME", raising=False)

    report = run_preflight(
        Backend.KAGGLE,
        project_root=tmp_path,
        env_file=_profile_env(tmp_path, include_token=False),
        kaggle_account="acc2",
    )

    issue = next(item for item in report.issues if item.check == "kaggle-credentials")
    assert "KAGGLE_ACC2_API_TOKEN" in issue.message
    assert "secondary-token" not in issue.message
