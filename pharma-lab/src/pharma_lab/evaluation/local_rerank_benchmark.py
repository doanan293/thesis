"""Choose the CPU llama-reranker configuration by measuring it end to end."""

from __future__ import annotations

import platform
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pharma_lab.artifacts.jsonl import iter_jsonl_objects
from pharma_lab.config.paths import COMPOSE_FILE, LOCAL_PROFILE_DIR
from pharma_lab.runtime.benchmarking import (
    LOCAL_RERANK_BENCHMARK_GROUPS,
    BenchmarkMeasurement,
    LevelResult,
    RerankGroup,
    check_score_consistency,
    describe_levels,
    percentile,
    recommend,
    rerank_groups_from_rows,
    sample_rerank_groups,
)
from pharma_lab.runtime.catalog import ModelSpec
from pharma_lab.runtime.compose import (
    LlamaCppComposeManager,
    compose_llama_cpp_image,
)
from pharma_lab.runtime.runtime_profiles import (
    RuntimeCandidate,
    RuntimeProfile,
    RuntimeProfileIdentity,
    RuntimeProfileStore,
    RuntimeSearchSpace,
    canonical_sha256,
)
from pharma_lab.runtime.server_policy import inference_cache_policy

CPUINFO_PATH = Path("/proc/cpuinfo")
LOCAL_TOPOLOGY = "cpu_compose"


class RerankServer(Protocol):
    def start(self, runtime: RuntimeCandidate) -> str: ...

    def logs(self) -> str: ...


class NativeRerankClient(Protocol):
    def rerank_native(
        self, query: str, documents: list[str], model: str
    ) -> list[float]: ...


@dataclass(frozen=True)
class ComposeRerankServer:
    """The compose llama-reranker service, recreated for every level."""

    manager: LlamaCppComposeManager
    spec: ModelSpec
    gguf_root: Path

    def start(self, runtime: RuntimeCandidate) -> str:
        return self.manager.ensure(
            "reranker", self.spec, self.gguf_root, runtime=runtime
        )

    def logs(self) -> str:
        return self.manager.logs("reranker")


@dataclass(frozen=True)
class LocalRerankBenchmarkResult:
    profile: RuntimeProfile
    path: Path
    measurements: tuple[BenchmarkMeasurement, ...]

    @property
    def selected_measurement(self) -> BenchmarkMeasurement:
        return next(
            item
            for item in self.measurements
            if item.status == "ok" and item.candidate == self.profile.selected
        )


def cpu_model_name(cpuinfo_path: Path = CPUINFO_PATH) -> str:
    try:
        text = cpuinfo_path.read_text(encoding="utf-8")
    except OSError:
        text = ""
    for line in text.splitlines():
        key, _separator, value = line.partition(":")
        if key.strip() == "model name" and value.strip():
            return value.strip()
    return platform.processor() or platform.machine() or "unknown-cpu"


def local_rerank_search_space(spec: ModelSpec) -> RuntimeSearchSpace:
    if spec.local_rerank_search_space is None:
        raise ValueError(f"{spec.name} has no local rerank search space")
    return spec.local_rerank_search_space


def local_rerank_profile_identity(
    spec: ModelSpec,
    *,
    compose_file: Path = COMPOSE_FILE,
    cpuinfo_path: Path = CPUINFO_PATH,
) -> RuntimeProfileIdentity:
    return RuntimeProfileIdentity.create(
        workload="rerank",
        model=spec.name,
        model_sha256=spec.sha256,
        runtime_sha256=canonical_sha256(
            {"image": compose_llama_cpp_image(compose_file)}
        ),
        inference_cache_policy_sha256=inference_cache_policy(spec).sha256,
        machine_shape=cpu_model_name(cpuinfo_path),
        topology=LOCAL_TOPOLOGY,
        search_space=local_rerank_search_space(spec),
    )


def load_local_rerank_profile(
    spec: ModelSpec,
    *,
    profile_root: Path = LOCAL_PROFILE_DIR,
    compose_file: Path = COMPOSE_FILE,
    cpuinfo_path: Path = CPUINFO_PATH,
) -> RuntimeProfile | None:
    identity = local_rerank_profile_identity(
        spec, compose_file=compose_file, cpuinfo_path=cpuinfo_path
    )
    return RuntimeProfileStore(profile_root).load(identity, model_slug=spec.slug)


def run_local_rerank_benchmark(
    *,
    spec: ModelSpec,
    candidate_data_path: Path,
    server: RerankServer,
    client_factory: Callable[[str], NativeRerankClient],
    profile_root: Path = LOCAL_PROFILE_DIR,
    compose_file: Path = COMPOSE_FILE,
    cpuinfo_path: Path = CPUINFO_PATH,
    clock: Callable[[], float] = time.monotonic,
) -> LocalRerankBenchmarkResult:
    space = local_rerank_search_space(spec)
    documents_per_group = space.candidates[0].request_batch_size
    groups = sample_rerank_groups(
        rerank_groups_from_rows(
            iter_jsonl_objects(candidate_data_path),
            documents_per_group=documents_per_group,
        ),
        LOCAL_RERANK_BENCHMARK_GROUPS,
    )
    if not groups:
        raise ValueError(
            f"{candidate_data_path} has no query with {documents_per_group} candidates"
        )
    measurements = check_score_consistency(
        [
            _measure_level(spec, runtime, groups, server, client_factory, clock)
            for runtime in space.candidates
        ]
    )
    selected = recommend(measurements, objective="latency")
    if selected is None:
        raise RuntimeError(
            "no local rerank level produced valid scores:\n"
            + describe_levels(measurements)
        )
    identity = local_rerank_profile_identity(
        spec, compose_file=compose_file, cpuinfo_path=cpuinfo_path
    )
    profile = RuntimeProfile.create(
        identity,
        selected,
        sample_count=sum(len(group.documents) for group in groups),
        measurements=[item.to_dict() for item in measurements],
        benchmark_job_sha256=canonical_sha256(
            {
                "identity_sha256": identity.sha256,
                "query_ids": [group.query_id for group in groups],
            }
        ),
    )
    path = RuntimeProfileStore(profile_root).save(profile, model_slug=spec.slug)
    return LocalRerankBenchmarkResult(profile, path, measurements)


def _measure_level(
    spec: ModelSpec,
    runtime: RuntimeCandidate,
    groups: Sequence[RerankGroup],
    server: RerankServer,
    client_factory: Callable[[str], NativeRerankClient],
    clock: Callable[[], float],
) -> LevelResult:
    latencies: list[float] = []
    scores: dict[str, float] = {}
    try:
        client = client_factory(server.start(runtime))
        # Warm-up: the first group once, untimed, after the service was recreated.
        client.rerank_native(groups[0].query, list(groups[0].documents), spec.name)
        for group in groups:
            started = clock()
            values = client.rerank_native(group.query, list(group.documents), spec.name)
            latencies.append(max(0.0, clock() - started))
            scores.update(zip(group.score_keys(), values, strict=True))
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        return LevelResult(
            BenchmarkMeasurement.invalid(
                runtime, type(exc).__name__, f"{exc}\n{_server_logs(server)}"
            ),
            {},
        )
    return LevelResult(
        BenchmarkMeasurement(
            runtime,
            sum(len(group.documents) for group in groups),
            sum(group.characters for group in groups),
            sum(latencies),
            latency_p50_seconds=percentile(latencies, 0.50),
            latency_p95_seconds=percentile(latencies, 0.95),
        ),
        scores,
    )


def _server_logs(server: RerankServer) -> str:
    try:
        return server.logs()
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        return f"server logs unavailable: {exc}"
