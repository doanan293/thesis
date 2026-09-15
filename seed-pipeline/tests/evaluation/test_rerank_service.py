import shutil
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.integrations.kaggle.factories import job_identity, stage_job

from seed_pipeline.artifacts.manifest import Completion
from seed_pipeline.cache.jsonl_records import append_record
from seed_pipeline.config.paths import rerank_log_path
from seed_pipeline.evaluation import rerank_service
from seed_pipeline.evaluation.local_rerank_benchmark import LocalRerankBenchmarkResult
from seed_pipeline.evaluation.rerank_cache_checkpoint import RerankCacheCheckpoint
from seed_pipeline.evaluation.rerank_score_cache import RerankScoreCache
from seed_pipeline.evaluation.rerank_service import (
    KaggleRerankBackend,
    LocalRerankBackend,
    RerankRequest,
    _cleanup_completed_stage_artifact,
)
from seed_pipeline.evaluation.run_workspace import load_run_record
from seed_pipeline.integrations.kaggle import auto_profile, kernel_service, sessions
from seed_pipeline.integrations.kaggle import service as kaggle_service
from seed_pipeline.integrations.kaggle.api import KaggleCommandRunner
from seed_pipeline.integrations.kaggle.config import (
    KaggleAccountProfile,
    OwnerConfiguration,
)
from seed_pipeline.integrations.kaggle.models import (
    ActionVerb,
    CloudArtifact,
    KernelPresence,
    KernelRemoteState,
    KernelStatus,
    ReconcileAction,
)
from seed_pipeline.integrations.kaggle.quota import AccountQuota
from seed_pipeline.integrations.kaggle.service import KaggleExecutionContext
from seed_pipeline.integrations.kaggle.workers.runtime import write_artifact_manifest
from seed_pipeline.runtime.benchmarking import BenchmarkMeasurement
from seed_pipeline.runtime.catalog import LOCAL_RERANK_SEARCH_SPACE, require_model
from seed_pipeline.runtime.runtime_profiles import (
    RuntimeProfile,
    RuntimeProfileIdentity,
)


@pytest.fixture(autouse=True)
def isolated_rerank_cache(tmp_path, monkeypatch):
    from seed_pipeline.config import paths

    monkeypatch.setattr(paths, "RERANK_SCORE_CACHE_DIR", tmp_path / "rerank-cache")


class FakeReranker:
    def rerank(self, query, candidates):
        del query
        return [
            candidate.with_rerank_score(float(index), index)
            for index, candidate in enumerate(candidates, start=1)
        ]


def request(run_root, model, *, force=False, dry_run=False):
    return RerankRequest(
        run_root=run_root,
        model=model,
        force=force,
        dry_run=dry_run,
        budget_seconds=60,
        request_timeout_seconds=5.0,
    )


def fake_local_backend():
    return LocalRerankBackend(reranker_factory=lambda _spec, _timeout: FakeReranker())


MODEL = "qwen3-reranker:0.6b-fp16"


def _session_contexts(*names: str) -> tuple[KaggleExecutionContext, ...]:
    return tuple(
        KaggleExecutionContext(
            KaggleAccountProfile(name, f"user-{name}", f"token-{name}"),
            OwnerConfiguration(
                f"user-{name}", "user-acc1", "user-acc1", f"user-{name}"
            ),
            KaggleCommandRunner(environment={}),
        )
        for name in names
    )


def _scores_artifact(cache: RerankScoreCache) -> CloudArtifact:
    return CloudArtifact(
        cache.path,
        cache.path.with_name("manifest.json"),
        job_identity(),
        Completion(1, 1, 0),
    )


@pytest.fixture
def kaggle_sessions(monkeypatch, tmp_path) -> dict[str, float]:
    """Fake Kaggle: GPU hours left per profile and a reusable runtime profile."""
    spec = require_model(MODEL)
    assert spec.rerank_search_space is not None
    selected = spec.rerank_search_space.candidates[0]
    quotas = {"acc1": 27.58, "acc2": 29.61}

    def read_quota(context: KaggleExecutionContext) -> AccountQuota:
        assert context.profile is not None
        return AccountQuota(
            context.profile.name,
            context.profile.username,
            quotas[context.profile.name],
            30.0,
            datetime(2026, 9, 19),
        )

    monkeypatch.setattr(rerank_service, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(
        kaggle_service,
        "resolve_session_contexts",
        lambda _account: _session_contexts(*quotas),
    )
    monkeypatch.setattr(
        kaggle_service, "runtime_manifest_sha256", lambda *_args, **_kwargs: "c" * 64
    )
    monkeypatch.setattr(
        kaggle_service, "active_kernel_profile", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        auto_profile,
        "ensure_runtime_profile",
        lambda **_kwargs: SimpleNamespace(
            profile=SimpleNamespace(selected=selected), action="reuse"
        ),
    )
    monkeypatch.setattr(sessions, "read_account_quota", read_quota)
    return quotas


def _score_record(*, query_id: str, score: float) -> dict[str, object]:
    return {
        "reranker": "qwen3-reranker:0.6b-fp16",
        "model_sha256": "model-sha",
        "request_contract_sha256": "contract-sha",
        "query_id": query_id,
        "query_hash": f"query-hash-{query_id}",
        "chunk_id": "chunk-1",
        "document_hash": "document-hash-1",
        "score": score,
    }


def test_remote_rerank_reconciliation_replaces_conflicts_and_keeps_local_only(
    tmp_path,
):
    """A missing replacement would keep conflicting local scores and fail merge."""
    local_path = tmp_path / "local.jsonl"
    remote_path = tmp_path / "remote.jsonl"
    append_record(
        local_path,
        _score_record(query_id="shared", score=0.1),
        schema="rerank-score-v2",
    )
    append_record(
        local_path,
        _score_record(query_id="local-only", score=0.2),
        schema="rerank-score-v2",
    )
    append_record(
        remote_path,
        _score_record(query_id="shared", score=0.9),
        schema="rerank-score-v2",
    )

    remote = RerankScoreCache(
        remote_path,
        model_sha256="model-sha",
        request_contract_sha256="contract-sha",
    )
    rerank_service._merge_remote_rerank_scores(
        local_path,
        remote,
        model_sha256="model-sha",
        request_contract_sha256="contract-sha",
    )

    reconciled = RerankScoreCache(
        local_path,
        model_sha256="model-sha",
        request_contract_sha256="contract-sha",
    )
    scores_by_query = {key.query_id: score for key, score in reconciled.records.items()}
    assert scores_by_query == {"shared": 0.9, "local-only": 0.2}


def test_two_models_register_two_variants_without_mutating_candidates(complete_run):
    candidates_path = complete_run / "candidates" / "candidates.jsonl"
    before = candidates_path.read_bytes()
    backend = fake_local_backend()

    first = backend.run(request(complete_run, "qwen3-reranker:0.6b-fp16"))
    second = backend.run(request(complete_run, "bge-reranker-v2-m3:f16"))

    record = load_run_record(complete_run / "run.json")
    assert len(record.rerank_variants) == 2
    assert first.artifact_dir is not None
    assert second.artifact_dir is not None
    assert first.artifact_dir != second.artifact_dir
    assert candidates_path.read_bytes() == before


def test_local_rerun_reuses_the_same_variant(complete_run):
    backend = fake_local_backend()
    first = backend.run(request(complete_run, "qwen3-reranker:0.6b-fp16"))
    assert first.artifact_dir is not None
    score_bytes = (first.artifact_dir / "rerank_scores.jsonl").read_bytes()

    second = backend.run(request(complete_run, "qwen3-reranker:0.6b-fp16"))

    assert second.artifact_dir is not None
    assert second.variant_sha256 == first.variant_sha256
    assert second.artifact_dir == first.artifact_dir
    assert (second.artifact_dir / "rerank_scores.jsonl").read_bytes() == score_bytes


def fake_kaggle_dry_run(**_kwargs):
    return SimpleNamespace(
        artifact_path=None,
        completion=Completion(total=1, complete=0, missing=1),
        actions=(
            ReconcileAction(
                "model", "owner/model", ActionVerb.CREATE, "dataset is missing"
            ),
            ReconcileAction(
                "input", "owner/input", ActionVerb.CREATE, "dataset is missing"
            ),
        ),
    )


def test_kaggle_dry_run_formats_resource_identity_and_does_not_register(
    complete_run, monkeypatch, tmp_path
):
    monkeypatch.setattr(kaggle_service, "run_kaggle_stage", fake_kaggle_dry_run)
    monkeypatch.setattr(
        kaggle_service,
        "resolve_session_contexts",
        lambda _account: _session_contexts("acc1"),
    )
    original_ensure_profile = auto_profile.ensure_runtime_profile

    def isolated_profile(**kwargs):
        return original_ensure_profile(
            **kwargs,
            profile_root=tmp_path / "profiles",
        )

    monkeypatch.setattr(auto_profile, "ensure_runtime_profile", isolated_profile)

    result = KaggleRerankBackend().run(
        request(complete_run, "qwen3-reranker:0.6b-fp16", dry_run=True)
    )

    assert result.incomplete
    assert "profile=benchmark-required" in result.actions
    assert load_run_record(complete_run / "run.json").rerank_variants == {}


def test_kaggle_rerank_allows_different_models_in_the_same_run(
    complete_run, monkeypatch, tmp_path
):
    from seed_pipeline.evaluation import rerank_service

    monkeypatch.setattr(rerank_service, "WORK_DIR", tmp_path / "work")

    first_entered = threading.Event()
    release_first = threading.Event()

    def run_unlocked(rerank_request):
        if rerank_request.model == "qwen3-reranker:0.6b-fp16":
            first_entered.set()
            assert release_first.wait(2)
        return rerank_request.model

    monkeypatch.setattr(
        KaggleRerankBackend,
        "_run_kaggle_unlocked",
        staticmethod(run_unlocked),
    )
    backend = KaggleRerankBackend()
    first_request = request(complete_run, "qwen3-reranker:0.6b-fp16")
    second_request = request(complete_run, "bge-reranker-v2-m3:f16")

    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(backend.run, first_request)
        assert first_entered.wait(2)
        try:
            second = backend.run(second_request)
        finally:
            release_first.set()

        assert first.result(timeout=2) == first_request.model
    assert second == second_request.model


def test_kaggle_rerank_serializes_same_model_across_runs(tmp_path, monkeypatch):
    from seed_pipeline.evaluation import rerank_service

    monkeypatch.setattr(rerank_service, "WORK_DIR", tmp_path / "work")

    first_entered = threading.Event()
    release_first = threading.Event()

    def run_unlocked(_request):
        first_entered.set()
        assert release_first.wait(2)
        return "first"

    monkeypatch.setattr(
        KaggleRerankBackend,
        "_run_kaggle_unlocked",
        staticmethod(run_unlocked),
    )
    backend = KaggleRerankBackend()
    first_request = request(tmp_path / "run-a", "qwen3-reranker:0.6b-fp16")
    second_request = request(tmp_path / "run-b", "qwen3-reranker:0.6b-fp16")

    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(backend.run, first_request)
        assert first_entered.wait(2)
        try:
            with pytest.raises(RuntimeError, match="already has an active Kaggle job"):
                backend.run(second_request)
        finally:
            release_first.set()
        assert first.result(timeout=2) == "first"


def test_completed_stage_cleanup_removes_only_job_directory(tmp_path):
    root = tmp_path / "kaggle-rerank-scores" / "model"
    job = root / "rerank" / "model" / "job-a"
    sibling = root / "rerank" / "model" / "job-b"
    artifact = job / "rerank_scores.jsonl"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("score\n", encoding="utf-8")
    sibling.mkdir(parents=True)
    (sibling / "keep.jsonl").write_text("keep\n", encoding="utf-8")

    _cleanup_completed_stage_artifact(artifact, root)

    assert not job.exists()
    assert (sibling / "keep.jsonl").is_file()
    assert root.is_dir()


def test_completed_stage_cleanup_rejects_artifact_outside_staging_root(tmp_path):
    root = tmp_path / "expected"
    artifact = tmp_path / "outside" / "rerank_scores.jsonl"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("score\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside Kaggle rerank staging root"):
        _cleanup_completed_stage_artifact(artifact, root)

    assert artifact.is_file()


def test_local_dry_run_reports_target_and_missing_pairs(complete_run):
    result = fake_local_backend().run(
        request(complete_run, "qwen3-reranker:0.6b-fp16", dry_run=True)
    )

    assert "missing_pairs=1" in result.actions
    assert (
        f"target={complete_run / 'rerank' / 'qwen3_reranker_0_6b_fp16'}"
        in result.actions
    )
    assert load_run_record(complete_run / "run.json").rerank_variants == {}


def test_forced_rerank_rescores_into_the_same_variant(complete_run):
    backend = fake_local_backend()
    first = backend.run(request(complete_run, "qwen3-reranker:0.6b-fp16"))

    second = backend.run(request(complete_run, "qwen3-reranker:0.6b-fp16", force=True))

    assert second.artifact_dir == first.artifact_dir
    assert "scored=1" in second.actions
    assert list(load_run_record(complete_run / "run.json").rerank_variants) == [
        "qwen3_reranker_0_6b_fp16"
    ]


def test_local_rerank_from_a_complete_cache_starts_no_reranker(
    complete_run, complete_rerank_cache, monkeypatch
):
    monkeypatch.setattr(
        rerank_service,
        "rerank_score_cache_path",
        lambda _model: complete_rerank_cache.path,
    )

    def no_server(_spec, _timeout):
        raise AssertionError("a complete cache must not start a reranker")

    result = LocalRerankBackend(reranker_factory=no_server).run(
        request(complete_run, "qwen3-reranker:0.6b-fp16")
    )

    assert result.artifact_dir == complete_run / "rerank" / "qwen3_reranker_0_6b_fp16"
    assert "scored=0" in result.actions


class RecordingReranker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query, candidates):
        self.calls.append((query, [candidate.chunk_id for candidate in candidates]))
        return [
            candidate.with_rerank_score(float(index), index)
            for index, candidate in enumerate(candidates, start=1)
        ]


def test_local_rerank_scores_each_query_in_one_request(two_candidate_run):
    reranker = RecordingReranker()

    result = LocalRerankBackend(reranker_factory=lambda _spec, _timeout: reranker).run(
        request(two_candidate_run, "qwen3-reranker:0.6b-fp16")
    )

    assert reranker.calls == [("test query", ["chunk-1", "chunk-2"])]
    assert "scored=2" in result.actions
    assert result.artifact_dir is not None


def test_local_benchmark_reports_the_selected_level_without_registering(
    complete_run,
):
    selected = LOCAL_RERANK_SEARCH_SPACE.candidates[2]
    measurement = BenchmarkMeasurement(
        selected, 90, 1000, 3.0, latency_p50_seconds=0.4, latency_p95_seconds=0.5
    )
    identity = RuntimeProfileIdentity.create(
        workload="rerank",
        model="qwen3-reranker:4b-fp16",
        model_sha256="a" * 64,
        runtime_sha256="b" * 64,
        inference_cache_policy_sha256="c" * 64,
        machine_shape="cpu",
        topology="cpu_compose",
        search_space=LOCAL_RERANK_SEARCH_SPACE,
    )
    profile = RuntimeProfile.create(
        identity,
        selected,
        sample_count=90,
        measurements=[measurement.to_dict()],
        benchmark_job_sha256="d" * 64,
    )
    report = Path("profiles/rerank/qwen3_reranker_4b_fp16.json")
    seen = []

    def fake_benchmark(spec, candidate_data_path, timeout):
        seen.append((spec.name, candidate_data_path, timeout))
        return LocalRerankBenchmarkResult(profile, report, (measurement,))

    def no_server(_spec, _timeout):
        raise AssertionError("a benchmark must not score the run")

    result = LocalRerankBackend(
        reranker_factory=no_server, benchmark_runner=fake_benchmark
    ).run(replace(request(complete_run, "qwen3-reranker:4b-fp16"), benchmark=True))

    assert seen == [
        (
            "qwen3-reranker:4b-fp16",
            complete_run / "candidates" / "candidates.jsonl",
            5.0,
        )
    ]
    assert result.actions == (
        "selected=server_slots=8 ubatch=4096 threads=8",
        "latency_p95_seconds=0.5",
        "env=LLAMA_RERANKER_CONTEXT_SIZE=20480 LLAMA_RERANKER_PARALLEL=8 "
        "LLAMA_RERANKER_THREADS=8 LLAMA_RERANKER_UBATCH_SIZE=4096",
    )
    assert (result.benchmark_levels, result.benchmark_report) == (1, report)
    assert load_run_record(complete_run / "run.json").rerank_variants == {}


def test_local_reranker_starts_compose_with_the_stored_local_profile(monkeypatch):
    spec = require_model("qwen3-reranker:4b-fp16")
    selected = LOCAL_RERANK_SEARCH_SPACE.candidates[2]
    ensured = []

    class FakeManager:
        def __init__(self, compose_file):
            self.compose_file = compose_file

        def ensure(self, role, ensured_spec, gguf_root, runtime=None):
            del gguf_root
            ensured.append((role, ensured_spec.name, runtime))
            return "http://127.0.0.1:11435"

    monkeypatch.setattr(
        rerank_service,
        "load_local_rerank_profile",
        lambda _spec: SimpleNamespace(selected=selected),
    )
    monkeypatch.setattr(rerank_service, "LlamaCppComposeManager", FakeManager)

    LocalRerankBackend._default_reranker(spec, 5.0)

    assert ensured == [("reranker", spec.name, selected)]


def test_kaggle_backend_leaves_benchmarks_to_the_runtime_profile(complete_run):
    with pytest.raises(ValueError, match="--backend local"):
        KaggleRerankBackend().run(
            replace(request(complete_run, "qwen3-reranker:0.6b-fp16"), benchmark=True)
        )


def test_kaggle_rerank_merges_session_scores_and_registers_the_variant(
    complete_run, complete_rerank_cache, kaggle_sessions, monkeypatch, tmp_path
):
    calls: list[dict] = []

    def fake_stage(**kwargs):
        calls.append(kwargs)
        kwargs["artifact_sink"](
            stage_job(tmp_path / "job"), _scores_artifact(complete_rerank_cache)
        )
        return SimpleNamespace(
            completion=Completion(1, 1, 0), actions=(), artifact_path=None
        )

    monkeypatch.setattr(kaggle_service, "run_kaggle_stage", fake_stage)

    result = KaggleRerankBackend().run(
        replace(
            request(complete_run, MODEL), kaggle_account="auto", budget_seconds=21_600
        )
    )

    assert not result.incomplete
    assert result.artifact_dir == complete_run / "rerank" / "qwen3_reranker_0_6b_fp16"
    assert [
        (
            call["kaggle_account"],
            call["budget_seconds"],
            call["max_runs"],
            call["force"],
            call["resume_remote"],
        )
        for call in calls
    ] == [("acc2", 21_600, 1, False, True)]
    assert isinstance(calls[0]["local_checkpoint"], RerankCacheCheckpoint)
    assert "stop=complete" in result.actions
    log = rerank_log_path(MODEL).read_text(encoding="utf-8")
    assert "quota account=acc1 remaining=27.58h" in log
    assert "session=1 start account=acc2" in log
    assert "merged artifact pairs=1/1" in log


def test_kaggle_rerank_stops_with_a_quota_table_when_no_account_has_an_hour(
    complete_run, kaggle_sessions, monkeypatch
):
    kaggle_sessions.update(acc1=1.2, acc2=0.4)

    def no_session(**_kwargs):
        raise AssertionError("no Kaggle session may start without quota")

    monkeypatch.setattr(kaggle_service, "run_kaggle_stage", no_session)

    result = KaggleRerankBackend().run(
        replace(
            request(complete_run, MODEL), kaggle_account="auto", budget_seconds=21_600
        )
    )

    assert result.incomplete
    assert "stop=quota-exhausted" in result.actions
    assert "missing_pairs=1" in result.actions
    assert len(result.quota) == 3
    assert "2026-09-19T00:00:00" in result.quota[1]
    assert load_run_record(complete_run / "run.json").rerank_variants == {}


def test_kaggle_rerank_attaches_to_the_account_already_running_the_kernel(
    complete_run, complete_rerank_cache, kaggle_sessions, monkeypatch, tmp_path
):
    accounts: list[str] = []

    def fake_stage(**kwargs):
        accounts.append(kwargs["kaggle_account"])
        kwargs["artifact_sink"](
            stage_job(tmp_path / "job"), _scores_artifact(complete_rerank_cache)
        )
        return SimpleNamespace(
            completion=Completion(1, 1, 0), actions=(), artifact_path=None
        )

    monkeypatch.setattr(
        kaggle_service, "active_kernel_profile", lambda *_args, **_kwargs: "acc1"
    )
    monkeypatch.setattr(kaggle_service, "run_kaggle_stage", fake_stage)

    KaggleRerankBackend().run(
        replace(
            request(complete_run, MODEL), kaggle_account="auto", budget_seconds=21_600
        )
    )

    assert accounts == ["acc1"]


def test_forced_kaggle_rerank_forces_only_the_first_session(
    complete_run, complete_rerank_cache, kaggle_sessions, monkeypatch, tmp_path
):
    calls: list[tuple[bool, bool]] = []

    def fake_stage(**kwargs):
        calls.append((kwargs["force"], kwargs["resume_remote"]))
        if len(calls) == 1:
            return SimpleNamespace(
                completion=Completion(1, 0, 1), actions=(), artifact_path=None
            )
        kwargs["artifact_sink"](
            stage_job(tmp_path / "job"), _scores_artifact(complete_rerank_cache)
        )
        return SimpleNamespace(
            completion=Completion(1, 1, 0), actions=(), artifact_path=None
        )

    monkeypatch.setattr(kaggle_service, "run_kaggle_stage", fake_stage)

    result = KaggleRerankBackend().run(
        replace(
            request(complete_run, MODEL, force=True),
            kaggle_account="auto",
            budget_seconds=21_600,
        )
    )

    assert calls == [(True, False), (False, False)]
    assert not result.incomplete


def test_kaggle_rerank_finalizes_a_complete_local_cache_without_kaggle(
    complete_run, complete_rerank_cache, monkeypatch, tmp_path
):
    def no_accounts(_account):
        raise AssertionError("a complete local cache needs no Kaggle account")

    monkeypatch.setattr(rerank_service, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(
        rerank_service,
        "rerank_score_cache_path",
        lambda _model: complete_rerank_cache.path,
    )
    monkeypatch.setattr(kaggle_service, "resolve_session_contexts", no_accounts)

    result = KaggleRerankBackend().run(request(complete_run, MODEL))

    assert result.artifact_dir == complete_run / "rerank" / "qwen3_reranker_0_6b_fp16"
    assert "reuse=local score cache" in result.actions


KERNEL = "user-acc2/rerank-5f22fcadeede1072"


def _fake_kernel_service(
    status: KernelStatus,
    write_output: Callable[[Path], None],
    seen: list[tuple[str, str]],
):
    class FakeKernelService:
        def __init__(self, runner, owner):
            self.owner = owner

        def inspect_state(self, reference):
            seen.append(("inspect", reference))
            return KernelRemoteState(reference, KernelPresence.EXISTS, status)

        def download_output(self, reference, destination):
            seen.append(("download", self.owner))
            write_output(Path(destination))

    return FakeKernelService


def _kernel_output(cache: RerankScoreCache, *, model: str = MODEL):
    def write_output(destination: Path) -> None:
        artifact_dir = destination / "artifact"
        artifact_dir.mkdir(parents=True)
        data = artifact_dir / "rerank_scores.jsonl"
        shutil.copy2(cache.path, data)
        write_artifact_manifest(
            data,
            artifact_type="rerank_scores",
            identity=job_identity(model=model),
            completion=Completion(300_000, 1, 299_999),
        )

    return write_output


@pytest.fixture
def recovery_accounts(monkeypatch, tmp_path):
    monkeypatch.setattr(rerank_service, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(
        kaggle_service,
        "resolve_profile_execution_contexts",
        lambda: _session_contexts("acc1", "acc2"),
    )


def test_recover_kernel_merges_its_scores_and_registers_a_complete_variant(
    complete_run, complete_rerank_cache, recovery_accounts, monkeypatch
):
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(
        kernel_service,
        "KernelService",
        _fake_kernel_service(
            KernelStatus.ERROR, _kernel_output(complete_rerank_cache), seen
        ),
    )

    result = KaggleRerankBackend().run(
        replace(request(complete_run, MODEL), recover_kernel=KERNEL)
    )

    assert seen == [("inspect", KERNEL), ("download", "user-acc2")]
    assert "recovered_pairs=1" in result.actions
    assert "missing_pairs=0" in result.actions
    assert result.artifact_dir == complete_run / "rerank" / "qwen3_reranker_0_6b_fp16"


def test_recover_kernel_refuses_a_running_kernel(
    complete_run, complete_rerank_cache, recovery_accounts, monkeypatch
):
    monkeypatch.setattr(
        kernel_service,
        "KernelService",
        _fake_kernel_service(
            KernelStatus.RUNNING, _kernel_output(complete_rerank_cache), []
        ),
    )

    with pytest.raises(ValueError, match="no finished output"):
        KaggleRerankBackend().run(
            replace(request(complete_run, MODEL), recover_kernel=KERNEL)
        )


def test_recover_kernel_rejects_scores_of_another_model(
    complete_run, complete_rerank_cache, recovery_accounts, monkeypatch
):
    monkeypatch.setattr(
        kernel_service,
        "KernelService",
        _fake_kernel_service(
            KernelStatus.COMPLETE,
            _kernel_output(complete_rerank_cache, model="bge-reranker-v2-m3:f16"),
            [],
        ),
    )

    with pytest.raises(ValueError, match="another model"):
        KaggleRerankBackend().run(
            replace(request(complete_run, MODEL), recover_kernel=KERNEL)
        )
    assert load_run_record(complete_run / "run.json").rerank_variants == {}


def test_recover_kernel_needs_a_profile_for_the_kernel_owner(
    complete_run, recovery_accounts
):
    with pytest.raises(ValueError, match="has username stranger"):
        KaggleRerankBackend().run(
            replace(request(complete_run, MODEL), recover_kernel="stranger/rerank-1")
        )


def test_local_backend_rejects_kernel_recovery(complete_run):
    with pytest.raises(ValueError, match="requires --backend kaggle"):
        fake_local_backend().run(
            replace(request(complete_run, MODEL), recover_kernel=KERNEL)
        )
