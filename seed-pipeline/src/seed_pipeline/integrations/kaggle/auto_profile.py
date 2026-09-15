from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from seed_pipeline.config.paths import KAGGLE_PROFILE_DIR, WORK_DIR
from seed_pipeline.runtime.benchmarking import BenchmarkMeasurement, describe_levels
from seed_pipeline.runtime.catalog import require_model
from seed_pipeline.runtime.runtime_profiles import (
    RuntimeCandidate,
    RuntimeProfile,
    RuntimeProfileIdentity,
    RuntimeProfileStore,
    RuntimeSearchSpace,
)
from seed_pipeline.runtime.server_policy import inference_cache_policy


@dataclass(frozen=True)
class ProfileResolution:
    profile: RuntimeProfile | None
    path: Path
    action: str


def _search_space(model: str, workload: str) -> RuntimeSearchSpace:
    spec = require_model(model)
    if workload == "rerank":
        space = spec.rerank_search_space
    elif workload == "query-embed":
        space = (
            spec.embedding_search_space.query if spec.embedding_search_space else None
        )
    elif workload == "corpus-embed":
        space = (
            spec.embedding_search_space.corpus if spec.embedding_search_space else None
        )
    else:
        raise ValueError(f"unsupported runtime profile workload: {workload}")
    if space is None:
        raise ValueError(f"model has no runtime search space for {workload}")
    return space


def _load_benchmark_selection(
    artifact_path: Path, search_space: RuntimeSearchSpace
) -> tuple[RuntimeCandidate, tuple[BenchmarkMeasurement, ...], str, int]:
    data_path = Path(artifact_path)
    manifest_path = data_path.with_name("manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = [
            json.loads(line)
            for line in data_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid benchmark artifact: {data_path}") from exc
    try:
        measurements = tuple(
            BenchmarkMeasurement.from_dict(row["measurement"]) for row in rows
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("benchmark measurement row is malformed") from exc
    runtime = manifest.get("runtime") or {}
    recommendation = runtime.get("recommendation")
    if recommendation is None:
        raise RuntimeError(
            "runtime benchmark found no valid level:\n" + describe_levels(measurements)
        )
    try:
        selected = RuntimeCandidate.from_dict(recommendation)
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("benchmark recommendation is malformed") from exc
    if selected not in search_space.candidates:
        raise RuntimeError("benchmark recommendation is not in the search space")
    if not any(
        item.status == "ok" and item.candidate == selected for item in measurements
    ):
        raise RuntimeError("benchmark recommendation has no successful measurement")
    identity = manifest.get("identity")
    if not isinstance(identity, dict) or not isinstance(
        identity.get("job_sha256"), str
    ):
        raise RuntimeError("benchmark manifest is missing job identity")
    sample_count = runtime.get("sample_count")
    if not isinstance(sample_count, int) or sample_count < 1:
        raise RuntimeError("benchmark manifest has no sample count")
    return selected, measurements, identity["job_sha256"], sample_count


def ensure_runtime_profile(
    *,
    workload: str,
    benchmark_stage: str,
    model: str,
    input_path: Path,
    gguf_root: Path,
    budget_seconds: int,
    dry_run: bool,
    force: bool,
    kaggle_account: str | None = None,
    profile_root: Path = KAGGLE_PROFILE_DIR,
    runtime_sha256: str | None = None,
    benchmark_runner: Callable[..., Any] | None = None,
) -> ProfileResolution:
    del force
    spec = require_model(model)
    if dry_run and runtime_sha256 is None:
        return ProfileResolution(
            None,
            Path(profile_root) / workload / f"{spec.slug}.json",
            "benchmark-required",
        )
    if runtime_sha256 is None:
        from seed_pipeline.integrations.kaggle.service import runtime_manifest_sha256

        runtime_sha256 = runtime_manifest_sha256(kaggle_account=kaggle_account)
    space = _search_space(model, workload)
    identity = RuntimeProfileIdentity.create(
        workload=workload,
        model=model,
        model_sha256=spec.sha256,
        runtime_sha256=runtime_sha256,
        inference_cache_policy_sha256=inference_cache_policy(spec).sha256,
        machine_shape="NvidiaTeslaT4",
        topology=spec.topology.value,
        search_space=space,
    )
    store = RuntimeProfileStore(profile_root)
    path = store.path(identity, model_slug=spec.slug)
    existing = store.load(identity, model_slug=spec.slug)
    if existing is not None:
        return ProfileResolution(existing, path, "reuse")
    if dry_run:
        return ProfileResolution(None, path, "benchmark-required")
    benchmark_output_dir = WORK_DIR / "kaggle-runtime-benchmarks" / spec.slug
    if benchmark_runner is None:
        from seed_pipeline.integrations.kaggle.models import StageName
        from seed_pipeline.integrations.kaggle.service import run_kaggle_stage

        def benchmark_runner(
            *, input_path: Path, gguf_root: Path, benchmark_items: int
        ):
            return run_kaggle_stage(
                stage=StageName(benchmark_stage),
                model=model,
                input_path=Path(input_path),
                output_dir=benchmark_output_dir,
                gguf_root=Path(gguf_root),
                force=False,
                check_only=False,
                budget_seconds=budget_seconds,
                benchmark_items=benchmark_items,
                kaggle_account=kaggle_account,
            )

    result = benchmark_runner(
        input_path=input_path,
        gguf_root=gguf_root,
        benchmark_items=512,
    )
    if result.artifact_path is None or not result.completion.is_complete:
        raise RuntimeError("runtime benchmark did not produce a complete artifact")
    selected, measurements, benchmark_job_sha256, sample_count = (
        _load_benchmark_selection(result.artifact_path, space)
    )
    profile = RuntimeProfile.create(
        identity,
        selected,
        sample_count=sample_count,
        measurements=[item.to_dict() for item in measurements],
        benchmark_job_sha256=benchmark_job_sha256,
    )
    saved = store.save(profile, model_slug=spec.slug)
    reloaded = store.load(identity, model_slug=spec.slug)
    if reloaded is None:
        raise RuntimeError(f"runtime profile failed validation after save: {saved}")
    if benchmark_output_dir.exists():
        shutil.rmtree(benchmark_output_dir, ignore_errors=True)
    benchmarks_root = benchmark_output_dir.parent
    if benchmarks_root.exists() and not any(benchmarks_root.iterdir()):
        with suppress(OSError):
            benchmarks_root.rmdir()
    return ProfileResolution(reloaded, saved, "created")
