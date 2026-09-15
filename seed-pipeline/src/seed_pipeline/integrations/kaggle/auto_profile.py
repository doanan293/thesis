from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from seed_pipeline.config.paths import KAGGLE_PROFILE_DIR, WORK_DIR
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
) -> tuple[RuntimeCandidate, tuple[dict[str, object], ...], str]:
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
    recommendation = (manifest.get("runtime") or {}).get("recommendation")
    if not isinstance(recommendation, dict):
        raise RuntimeError("benchmark has no runtime recommendation")
    try:
        batch_size = int(recommendation["batch_size"])
        concurrency = int(recommendation["concurrency"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("benchmark recommendation is malformed") from exc
    candidates = [
        candidate
        for candidate in search_space.candidates
        if candidate.request_batch_size == batch_size
        and candidate.concurrency == concurrency
    ]
    if len(candidates) != 1:
        raise RuntimeError("benchmark recommendation does not identify one candidate")
    selected = candidates[0]
    measurements: list[dict[str, object]] = []
    for row in rows:
        measurement = row.get("measurement") if isinstance(row, dict) else None
        if not isinstance(measurement, dict):
            raise RuntimeError("benchmark measurement row is malformed")
        level = measurement.get("level")
        if not isinstance(level, dict):
            raise RuntimeError("benchmark measurement level is malformed")
        try:
            level_batch = int(level["batch_size"])
            level_concurrency = int(level["concurrency"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("benchmark measurement level is malformed") from exc
        matching = [
            candidate
            for candidate in search_space.candidates
            if candidate.request_batch_size == level_batch
            and candidate.concurrency == level_concurrency
        ]
        if len(matching) != 1:
            raise RuntimeError("benchmark measurement does not identify one candidate")
        measurements.append(dict(measurement) | {"candidate": matching[0].to_dict()})
    if not any(
        item.get("status") == "ok" and item.get("candidate") == selected.to_dict()
        for item in measurements
    ):
        raise RuntimeError("benchmark recommendation has no successful measurement")
    identity = manifest.get("identity")
    if not isinstance(identity, dict) or not isinstance(
        identity.get("job_sha256"), str
    ):
        raise RuntimeError("benchmark manifest is missing job identity")
    job_sha256 = identity["job_sha256"]
    return selected, tuple(measurements), job_sha256


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
    selected, measurements, benchmark_job_sha256 = _load_benchmark_selection(
        result.artifact_path, space
    )
    profile = RuntimeProfile.create(
        identity,
        selected,
        sample_count=512,
        measurements=measurements,
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
