import json
from pathlib import Path
from types import SimpleNamespace

from corpus_pipeline.artifacts.manifest import Completion
from corpus_pipeline.runtime.catalog import require_model
from corpus_pipeline.runtime.runtime_profiles import RuntimeProfileStore
from corpus_pipeline.integrations.kaggle.auto_profile import (
    ensure_runtime_profile,
)


MODEL = "qwen3-reranker:0.6b-fp16"


def _benchmark_result(tmp_path: Path):
    selected = require_model(MODEL).rerank_search_space.candidates[-1]
    data = tmp_path / "benchmark_results.jsonl"
    data.write_text(
        json.dumps(
            {
                "measurement": {
                    "level": {
                        "batch_size": selected.request_batch_size,
                        "concurrency": selected.concurrency,
                    },
                    "items": 512,
                    "input_characters": 100,
                    "elapsed_seconds": 1.0,
                    "status": "ok",
                    "error_category": None,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "runtime": {
                    "recommendation": {
                        "batch_size": selected.request_batch_size,
                        "concurrency": selected.concurrency,
                    }
                },
                "identity": {"job_sha256": "a" * 64},
            }
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(
        artifact_path=data,
        completion=Completion(1, 1, 0),
    )


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
    assert result.profile.sample_count == 512
    assert result.path.is_file()
    assert len(calls) == 1


def test_runtime_lookup_and_benchmark_use_selected_account(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(
        "corpus_pipeline.integrations.kaggle.service.runtime_manifest_sha256",
        lambda *, kaggle_account: seen.append(("runtime", kaggle_account)) or "b" * 64,
    )
    monkeypatch.setattr(
        "corpus_pipeline.integrations.kaggle.service.run_kaggle_stage",
        lambda **kwargs: seen.append(("benchmark", kwargs["kaggle_account"]))
        or _benchmark_result(tmp_path),
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
    monkeypatch.setattr("corpus_pipeline.integrations.kaggle.auto_profile.WORK_DIR", work_dir)
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
