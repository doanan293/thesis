import json
import os
from pathlib import Path

import pytest
from tests.integrations.kaggle.factories import rerank_runtime_profile

from seed_pipeline.integrations.kaggle import service as kaggle_service
from seed_pipeline.integrations.kaggle.api import KaggleCommandRunner
from seed_pipeline.integrations.kaggle.checkpoint_inheritance import (
    CheckpointInheritanceService,
)
from seed_pipeline.integrations.kaggle.checkpoints import CheckpointState
from seed_pipeline.integrations.kaggle.dataset_service import DatasetService
from seed_pipeline.integrations.kaggle.dependencies import DependencyService
from seed_pipeline.integrations.kaggle.models import (
    CloudArtifact,
    KernelPresence,
    KernelRemoteState,
    KernelStatus,
    StageJob,
    StageName,
)
from seed_pipeline.integrations.kaggle.service import (
    resolve_execution_context,
    resolve_profile_execution_contexts,
    resolve_session_contexts,
)

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
    environment = context.runner.environment
    assert environment is not None
    assert environment["KAGGLE_USERNAME"] == "secondary-user"
    assert environment["KAGGLE_API_TOKEN"] == "secondary-token"
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


def test_profile_contexts_are_numeric_and_account_isolated(tmp_path, monkeypatch):
    env_file = _write_profiles(tmp_path)
    _clear_kaggle_environment(monkeypatch)

    contexts = resolve_profile_execution_contexts(env_file=env_file)
    profiles = [item.profile for item in contexts]
    environments = [item.runner.environment for item in contexts]

    assert [profile.name if profile else None for profile in profiles] == [
        "acc1",
        "acc2",
    ]
    primary, secondary = environments
    assert primary is not None and secondary is not None
    assert primary["KAGGLE_USERNAME"] == "primary-user"
    assert secondary["KAGGLE_USERNAME"] == "secondary-user"
    assert not any(
        key.startswith("KAGGLE_ACC")
        for environment in (primary, secondary)
        for key in environment
    )
    assert "primary-token" not in secondary.values()
    assert "secondary-token" not in primary.values()


def test_profile_contexts_fail_before_runner_commands_for_incomplete_profile(
    tmp_path, monkeypatch
):
    env_file = _write_profiles(tmp_path)
    env_file.write_text(
        env_file.read_text(encoding="utf-8").replace(
            "KAGGLE_ACC2_API_TOKEN=secondary-token\n", ""
        ),
        encoding="utf-8",
    )
    _clear_kaggle_environment(monkeypatch)
    called = []

    def fail_run(*args, **kwargs):
        called.append((args, kwargs))
        raise AssertionError("runner command must not run")

    monkeypatch.setattr(kaggle_service.KaggleCommandRunner, "run", fail_run)
    with pytest.raises(ValueError, match="KAGGLE_ACC2_API_TOKEN"):
        resolve_profile_execution_contexts(env_file=env_file)
    assert called == []


def test_run_stage_uses_one_account_bound_context(tmp_path, monkeypatch):
    env_file = _write_profiles(tmp_path)
    _clear_kaggle_environment(monkeypatch)
    captured = {}
    sentinel = object()

    class FakeOrchestrator:
        def run(self, request):
            captured["request"] = request
            return sentinel

    def fake_make_orchestrator(owners, runner, **kwargs):
        captured["owners"] = owners
        captured["runner"] = runner
        captured["kwargs"] = kwargs
        return FakeOrchestrator()

    monkeypatch.setattr(kaggle_service, "make_orchestrator", fake_make_orchestrator)
    result = kaggle_service.run_kaggle_stage(
        stage=StageName.RERANK,
        model=MODEL,
        input_path=tmp_path / "candidates.jsonl",
        output_dir=tmp_path / "output",
        runtime_profile=rerank_runtime_profile(MODEL),
        kaggle_account="acc2",
        env_file=env_file,
    )

    assert result is sentinel
    assert captured["owners"].execution == "secondary-user"
    assert captured["runner"].environment["KAGGLE_USERNAME"] == "secondary-user"
    assert captured["request"].owners == captured["owners"]
    assert captured["kwargs"]["target_profile"] == "acc2"
    assert [
        context.profile.name for context in captured["kwargs"]["checkpoint_contexts"]
    ] == [
        "acc1",
        "acc2",
    ]


def test_make_orchestrator_binds_each_checkpoint_service_to_its_profile_runner(
    tmp_path, monkeypatch
):
    env_file = _write_profiles(tmp_path)
    _clear_kaggle_environment(monkeypatch)
    contexts = resolve_profile_execution_contexts(env_file=env_file)
    target = contexts[1]
    orchestrator = kaggle_service.make_orchestrator(
        target.owners,
        target.runner,
        target_profile="acc2",
        checkpoint_contexts=contexts,
        temp_root=tmp_path / "tmp",
    )

    inheritance = orchestrator.checkpoint_inheritance
    assert isinstance(inheritance, CheckpointInheritanceService)
    assert inheritance.target is orchestrator.checkpoints
    assert [item.profile_name for item in inheritance.candidates] == ["acc1", "acc2"]
    for item in inheritance.candidates:
        datasets = item.checkpoints.datasets
        assert isinstance(datasets, DatasetService)
        runner = datasets.runner
        assert isinstance(runner, KaggleCommandRunner)
        environment = runner.environment
        assert environment is not None
        assert environment["KAGGLE_USERNAME"] in {"primary-user", "secondary-user"}
        assert not any(key.startswith("KAGGLE_ACC") for key in environment)
        token = "primary-token" if item.profile_name == "acc1" else "secondary-token"
        assert token not in runner.redact(token)


def test_make_orchestrator_lets_every_profile_publish_its_own_datasets(
    tmp_path, monkeypatch
):
    env_file = _write_profiles(tmp_path)
    _clear_kaggle_environment(monkeypatch)
    contexts = resolve_profile_execution_contexts(env_file=env_file)
    target = contexts[1]

    orchestrator = kaggle_service.make_orchestrator(
        target.owners,
        target.runner,
        target_profile="acc2",
        checkpoint_contexts=contexts,
        temp_root=tmp_path / "tmp",
    )

    dependency_service = orchestrator.dependencies
    assert isinstance(dependency_service, DependencyService)
    assert sorted(dependency_service.publishers) == ["primary-user", "secondary-user"]
    for owner, publisher in dependency_service.publishers.items():
        runner = publisher.runner
        assert isinstance(runner, KaggleCommandRunner)
        assert runner.environment is not None
        assert runner.environment["KAGGLE_USERNAME"].casefold() == owner


class EmptyLocalSource:
    def inspect(self, job: StageJob, *, download_root: Path) -> CheckpointState:
        raise AssertionError("not used by this test")


def _ignore_artifact(job: StageJob, artifact: CloudArtifact) -> None:
    del job, artifact


def test_run_stage_passes_session_hooks_to_the_orchestrator(tmp_path, monkeypatch):
    env_file = _write_profiles(tmp_path)
    _clear_kaggle_environment(monkeypatch)
    captured = {}
    source = EmptyLocalSource()

    class FakeOrchestrator:
        def run(self, request):
            captured["request"] = request
            return "result"

    def fake_make_orchestrator(owners, runner, **kwargs):
        captured["kwargs"] = kwargs
        return FakeOrchestrator()

    monkeypatch.setattr(kaggle_service, "make_orchestrator", fake_make_orchestrator)

    kaggle_service.run_kaggle_stage(
        stage=StageName.RERANK,
        model=MODEL,
        input_path=tmp_path / "candidates.jsonl",
        output_dir=tmp_path / "output",
        runtime_profile=rerank_runtime_profile(MODEL),
        kaggle_account="acc2",
        env_file=env_file,
        max_runs=1,
        resume_remote=False,
        artifact_sink=_ignore_artifact,
        local_checkpoint=source,
    )

    assert captured["request"].max_runs == 1
    assert captured["request"].resume_remote is False
    assert captured["kwargs"]["artifact_sink"] is _ignore_artifact
    assert captured["kwargs"]["local_checkpoint"] is source


def test_make_orchestrator_wires_the_local_source_and_sink(tmp_path, monkeypatch):
    env_file = _write_profiles(tmp_path)
    _clear_kaggle_environment(monkeypatch)
    contexts = resolve_profile_execution_contexts(env_file=env_file)
    source = EmptyLocalSource()

    orchestrator = kaggle_service.make_orchestrator(
        contexts[0].owners,
        contexts[0].runner,
        target_profile="acc1",
        checkpoint_contexts=contexts,
        temp_root=tmp_path / "tmp",
        local_checkpoint=source,
        artifact_sink=_ignore_artifact,
    )

    inheritance = orchestrator.checkpoint_inheritance
    assert isinstance(inheritance, CheckpointInheritanceService)
    assert inheritance.local is source
    assert orchestrator.artifact_sink is _ignore_artifact


def test_session_contexts_for_auto_are_every_profile(tmp_path, monkeypatch):
    env_file = _write_profiles(tmp_path)
    _clear_kaggle_environment(monkeypatch)

    contexts = resolve_session_contexts("auto", env_file=env_file)

    assert [c.profile.name for c in contexts if c.profile] == ["acc1", "acc2"]


@pytest.mark.parametrize(("account", "expected"), [("acc2", "acc2"), (None, "acc1")])
def test_session_contexts_for_one_account(tmp_path, monkeypatch, account, expected):
    env_file = _write_profiles(tmp_path)
    _clear_kaggle_environment(monkeypatch)

    contexts = resolve_session_contexts(account, env_file=env_file)

    assert [c.profile.name for c in contexts if c.profile] == [expected]


def test_session_contexts_require_account_profiles(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "KAGGLE_USERNAME=legacy-user\nKAGGLE_API_TOKEN=legacy-token\n",
        encoding="utf-8",
    )
    _clear_kaggle_environment(monkeypatch)

    with pytest.raises(ValueError, match="need account profiles"):
        resolve_session_contexts(None, env_file=env_file)


def test_active_kernel_profile_finds_the_account_running_the_job(tmp_path, monkeypatch):
    env_file = _write_profiles(tmp_path)
    _clear_kaggle_environment(monkeypatch)
    contexts = resolve_profile_execution_contexts(env_file=env_file)
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "query": "query",
                "candidates": [{"chunk_id": "c1", "document_text": "text"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    inspected: list[str] = []

    class FakeKernelService:
        def __init__(self, runner, owner):
            self.owner = owner

        def inspect_state(self, reference):
            inspected.append(reference)
            status = (
                KernelStatus.RUNNING
                if self.owner == "secondary-user"
                else KernelStatus.COMPLETE
            )
            return KernelRemoteState(reference, KernelPresence.EXISTS, status)

    monkeypatch.setattr(kaggle_service, "KernelService", FakeKernelService)

    profile = kaggle_service.active_kernel_profile(
        contexts,
        stage=StageName.RERANK,
        model=MODEL,
        input_path=candidates,
        output_dir=tmp_path / "output",
        runtime_profile=rerank_runtime_profile(MODEL),
    )

    assert profile == "acc2"
    assert [reference.split("/")[0] for reference in inspected] == [
        "primary-user",
        "secondary-user",
    ]
