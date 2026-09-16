import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.integrations.kaggle.factories import rerank_runtime_profile

from pharma_lab.artifacts.manifest import Completion
from pharma_lab.integrations.kaggle.auto_profile import (
    ensure_runtime_profile,
)
from pharma_lab.runtime.benchmarking import BenchmarkMeasurement
from pharma_lab.runtime.catalog import require_model
from pharma_lab.runtime.server_policy import (
    InferenceCachePolicy,
)

MODEL = "qwen3-reranker:0.6b-fp16"


def _benchmark_result(
    tmp_path: Path, *, valid: bool = True, log_tail: str | None = None
):
    selected = rerank_runtime_profile(MODEL, index=-1)
    measurement = (
        BenchmarkMeasurement(
            selected, 960, 100, 1.0, latency_p50_seconds=0.1, latency_p95_seconds=0.2
        )
        if valid
        else BenchmarkMeasurement.invalid(selected, "ModelServerExited", log_tail)
    )
    data = tmp_path / "benchmark_results.jsonl"
    data.write_text(
        json.dumps({"measurement": measurement.to_dict()}) + "\n", encoding="utf-8"
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "runtime": {
                    "recommendation": selected.to_dict() if valid else None,
                    "sample_count": 960,
                },
                "identity": {"job_sha256": "a" * 64},
            }
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(artifact_path=data, completion=Completion(1, 1, 0))


def test_cache_miss_benchmarks_and_persists_selected_profile(tmp_path):
    calls = []

    def benchmark_runner(**kwargs):
        calls.append(kwargs)
        return _benchmark_result(tmp_path)

    result = ensure_runtime_profile(
        workload="rerank",
        benchmark_stage="rerank-benchmark",
        model=MODEL,
        input_path=tmp_path / "candidates.jsonl",
        gguf_root=tmp_path / "gguf",
        budget_seconds=60,
        dry_run=False,
        force=False,
        profile_root=tmp_path / "profiles",
        runtime_sha256="b" * 64,
        benchmark_runner=benchmark_runner,
    )

    assert result.action == "created"
    assert result.profile is not None
    assert result.profile.sample_count == 960
    assert result.path.is_file()
    assert len(calls) == 1


def test_runtime_lookup_and_benchmark_use_selected_account(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(
        "pharma_lab.integrations.kaggle.service.runtime_manifest_sha256",
        lambda *, kaggle_account: seen.append(("runtime", kaggle_account)) or "b" * 64,
    )
    monkeypatch.setattr(
        "pharma_lab.integrations.kaggle.service.run_kaggle_stage",
        lambda **kwargs: (
            seen.append(("benchmark", kwargs["kaggle_account"]))
            or _benchmark_result(tmp_path)
        ),
    )

    ensure_runtime_profile(
        workload="rerank",
        benchmark_stage="rerank-benchmark",
        model=MODEL,
        input_path=tmp_path / "candidates.jsonl",
        gguf_root=tmp_path / "gguf",
        budget_seconds=60,
        dry_run=False,
        force=False,
        profile_root=tmp_path / "profiles",
        kaggle_account="acc2",
    )

    assert seen == [("runtime", "acc2"), ("benchmark", "acc2")]


def test_cache_hit_skips_benchmark(tmp_path):
    first = ensure_runtime_profile(
        workload="rerank",
        benchmark_stage="rerank-benchmark",
        model=MODEL,
        input_path=tmp_path / "first.jsonl",
        gguf_root=tmp_path / "gguf",
        budget_seconds=60,
        dry_run=False,
        force=False,
        profile_root=tmp_path / "profiles",
        runtime_sha256="b" * 64,
        benchmark_runner=lambda **kwargs: _benchmark_result(tmp_path),
    )

    result = ensure_runtime_profile(
        workload="rerank",
        benchmark_stage="rerank-benchmark",
        model=MODEL,
        input_path=tmp_path / "different.jsonl",
        gguf_root=tmp_path / "gguf",
        budget_seconds=60,
        dry_run=False,
        force=True,
        profile_root=tmp_path / "profiles",
        runtime_sha256="b" * 64,
        benchmark_runner=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("benchmark must be skipped")
        ),
    )

    assert result.action == "reuse"
    assert result.profile == first.profile


def test_inference_cache_policy_change_invalidates_cached_profile(
    tmp_path, monkeypatch
):
    import pharma_lab.integrations.kaggle.auto_profile as auto_profile

    profile_root = tmp_path / "profiles"
    common = {
        "workload": "rerank",
        "benchmark_stage": "rerank-benchmark",
        "model": MODEL,
        "input_path": tmp_path / "candidates.jsonl",
        "gguf_root": tmp_path / "gguf",
        "budget_seconds": 60,
        "dry_run": False,
        "force": False,
        "profile_root": profile_root,
        "runtime_sha256": "b" * 64,
    }
    original = auto_profile.inference_cache_policy
    monkeypatch.setattr(
        auto_profile,
        "inference_cache_policy",
        lambda _spec: InferenceCachePolicy(8192, True, "query-adjacent-v1"),
    )
    ensure_runtime_profile(
        **common,
        benchmark_runner=lambda **_kwargs: _benchmark_result(tmp_path),
    )
    calls = []
    monkeypatch.setattr(auto_profile, "inference_cache_policy", original)

    result = ensure_runtime_profile(
        **common,
        benchmark_runner=lambda **kwargs: (
            calls.append(kwargs) or _benchmark_result(tmp_path)
        ),
    )

    assert len(calls) == 1
    assert result.action == "created"
    assert result.profile is not None
    assert result.profile.identity.payload["inference_cache_policy_sha256"] == (
        original(require_model(MODEL)).sha256
    )
    saved = list(profile_root.rglob("*.json"))
    assert len(saved) == 1
    assert (
        json.loads(saved[0].read_text(encoding="utf-8"))["identity"][
            "inference_cache_policy_sha256"
        ]
        == original(require_model(MODEL)).sha256
    )


def test_dry_run_cache_miss_does_not_benchmark_or_write(tmp_path):
    result = ensure_runtime_profile(
        workload="rerank",
        benchmark_stage="rerank-benchmark",
        model=MODEL,
        input_path=tmp_path / "candidates.jsonl",
        gguf_root=tmp_path / "gguf",
        budget_seconds=60,
        dry_run=True,
        force=False,
        profile_root=tmp_path / "profiles",
        runtime_sha256="b" * 64,
        benchmark_runner=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("dry-run must not benchmark")
        ),
    )

    assert result.action == "benchmark-required"
    assert result.profile is None
    assert not list((tmp_path / "profiles").rglob("*.json"))


def test_benchmark_cleans_up_work_directory(tmp_path, monkeypatch):
    work_dir = tmp_path / "work"
    monkeypatch.setattr(
        "pharma_lab.integrations.kaggle.auto_profile.WORK_DIR", work_dir
    )
    spec = require_model(MODEL)
    benchmark_dir = work_dir / "kaggle-runtime-benchmarks" / spec.slug
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    (benchmark_dir / "temp.log").write_text("log content", encoding="utf-8")

    result = ensure_runtime_profile(
        workload="rerank",
        benchmark_stage="rerank-benchmark",
        model=MODEL,
        input_path=tmp_path / "candidates.jsonl",
        gguf_root=tmp_path / "gguf",
        budget_seconds=60,
        dry_run=False,
        force=False,
        profile_root=tmp_path / "profiles",
        runtime_sha256="b" * 64,
        benchmark_runner=lambda **kwargs: _benchmark_result(tmp_path),
    )

    assert result.action == "created"
    assert not benchmark_dir.exists()
    assert not (work_dir / "kaggle-runtime-benchmarks").exists()


def test_profile_stores_the_recommended_candidate(tmp_path):
    result = ensure_runtime_profile(
        workload="rerank",
        benchmark_stage="rerank-benchmark",
        model=MODEL,
        input_path=tmp_path / "candidates.jsonl",
        gguf_root=tmp_path / "gguf",
        budget_seconds=60,
        dry_run=False,
        force=False,
        profile_root=tmp_path / "profiles",
        runtime_sha256="b" * 64,
        benchmark_runner=lambda **_kwargs: _benchmark_result(tmp_path),
    )

    assert result.profile is not None
    assert result.profile.selected == rerank_runtime_profile(MODEL, index=-1)
    assert result.profile.measurements[0]["candidate"] == (
        result.profile.selected.to_dict()
    )


def test_benchmark_without_a_valid_level_stops_with_the_server_log_tail(tmp_path):
    with pytest.raises(RuntimeError, match="CUDA error: out of memory"):
        ensure_runtime_profile(
            workload="rerank",
            benchmark_stage="rerank-benchmark",
            model=MODEL,
            input_path=tmp_path / "candidates.jsonl",
            gguf_root=tmp_path / "gguf",
            budget_seconds=60,
            dry_run=False,
            force=False,
            profile_root=tmp_path / "profiles",
            runtime_sha256="b" * 64,
            benchmark_runner=lambda **_kwargs: _benchmark_result(
                tmp_path, valid=False, log_tail="CUDA error: out of memory"
            ),
        )

    assert not list((tmp_path / "profiles").rglob("*.json"))
