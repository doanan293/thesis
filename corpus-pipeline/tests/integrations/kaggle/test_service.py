import os
from pathlib import Path

from corpus_pipeline.integrations.kaggle import service as kaggle_service
from corpus_pipeline.integrations.kaggle.models import StageName
from corpus_pipeline.integrations.kaggle.service import resolve_execution_context
from corpus_pipeline.runtime.catalog import require_model


MODEL = "qwen3-reranker:0.6b-fp16"


def _write_profiles(tmp_path: Path) -> Path:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "KAGGLE_ACCOUNT_DEFAULT=acc1\n"
        "KAGGLE_SHARED_OWNER=primary-user\n"
        "KAGGLE_ACC1_USERNAME=primary-user\n"
        "KAGGLE_ACC1_API_TOKEN=primary-token\n"
        "KAGGLE_ACC2_USERNAME=secondary-user\n"
        "KAGGLE_ACC2_API_TOKEN=secondary-token\n",
        encoding="utf-8",
    )
    return env_file


def _clear_kaggle_environment(monkeypatch):
    for key in tuple(os.environ):
        if key.startswith("KAGGLE_"):
            monkeypatch.delenv(key, raising=False)


def test_execution_context_pairs_acc2_runner_and_owners(tmp_path, monkeypatch):
    env_file = _write_profiles(tmp_path)
    _clear_kaggle_environment(monkeypatch)

    context = resolve_execution_context("acc2", env_file=env_file)

    assert context.profile is not None
    assert context.profile.name == "acc2"
    assert context.owners.execution == "secondary-user"
    assert context.owners.checkpoint == "secondary-user"
    assert context.owners.runtime == "primary-user"
    assert context.runner.environment["KAGGLE_USERNAME"] == "secondary-user"
    assert context.runner.environment["KAGGLE_API_TOKEN"] == "secondary-token"
    assert context.runner.redact("failed secondary-token") == "failed <redacted>"


def test_legacy_context_keeps_generic_credential_path(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "KAGGLE_USERNAME=legacy-user\nKAGGLE_API_TOKEN=legacy-token\n",
        encoding="utf-8",
    )
    _clear_kaggle_environment(monkeypatch)

    context = resolve_execution_context(None, env_file=env_file)

    assert context.profile is None
    assert context.owners.execution == "legacy-user"
    assert context.runner.environment is None


def test_run_stage_uses_one_account_bound_context(tmp_path, monkeypatch):
    env_file = _write_profiles(tmp_path)
    _clear_kaggle_environment(monkeypatch)
    captured = {}
    sentinel = object()

    class FakeOrchestrator:
        def run(self, request):
            captured["request"] = request
            return sentinel

    def fake_make_orchestrator(owners, runner):
        captured["owners"] = owners
        captured["runner"] = runner
        return FakeOrchestrator()

    monkeypatch.setattr(kaggle_service, "make_orchestrator", fake_make_orchestrator)
    result = kaggle_service.run_kaggle_stage(
        stage=StageName.RERANK,
        model=MODEL,
        input_path=tmp_path / "candidates.jsonl",
        output_dir=tmp_path / "output",
        runtime_profile=require_model(MODEL).rerank_search_space.candidates[0],
        kaggle_account="acc2",
        env_file=env_file,
    )

    assert result is sentinel
    assert captured["owners"].execution == "secondary-user"
    assert captured["runner"].environment["KAGGLE_USERNAME"] == "secondary-user"
    assert captured["request"].owners == captured["owners"]
