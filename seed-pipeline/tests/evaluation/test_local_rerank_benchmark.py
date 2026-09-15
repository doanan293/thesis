import json
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from seed_pipeline.evaluation.local_rerank_benchmark import (
    ComposeRerankServer,
    LocalRerankBenchmarkResult,
    cpu_model_name,
    load_local_rerank_profile,
    run_local_rerank_benchmark,
)
from seed_pipeline.runtime.catalog import (
    LOCAL_RERANK_SEARCH_SPACE,
    ModelSpec,
    require_model,
)
from seed_pipeline.runtime.compose import LlamaCppComposeManager, file_sha256

CPU = "13th Gen Intel(R) Core(TM) i5-13420H"
MODEL = "qwen3-reranker:4b-fp16"


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class RecordingRunner:
    """docker compose stand-in: records every `up` environment, can fail some levels."""

    def __init__(
        self, *, failing_ubatches: frozenset[str] = frozenset(), logs: str = ""
    ):
        self.ups: list[dict[str, str]] = []
        self.failing_ubatches = failing_ubatches
        self.logs = logs

    def run(self, args, env=None, capture_output=False):
        if capture_output:
            return self.logs
        environment = dict(env or {})
        self.ups.append(environment)
        if environment.get("LLAMA_RERANKER_UBATCH_SIZE") in self.failing_ubatches:
            raise subprocess.CalledProcessError(1, list(args))
        return ""


class HealthyClient:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint

    def health(self) -> None:
        return None


class ScoringClient:
    """Scores by document length; latency and drift depend on the level being served."""

    def __init__(
        self,
        runner: RecordingRunner,
        clock: FakeClock,
        *,
        drifting_ubatch: str | None = None,
    ) -> None:
        self.runner = runner
        self.clock = clock
        self.drifting_ubatch = drifting_ubatch
        self.calls: list[int] = []

    def rerank_native(
        self, query: str, documents: list[str], model: str
    ) -> list[float]:
        del query, model
        level = self.runner.ups[-1]
        self.calls.append(len(documents))
        threads = int(level["LLAMA_RERANKER_THREADS"])
        ubatch = level["LLAMA_RERANKER_UBATCH_SIZE"]
        # ubatch 4096 with 12 threads is the fastest level.
        self.clock.now += (1.0 if ubatch == "4096" else 2.0) * 8 / threads
        drift = 0.01 if ubatch == self.drifting_ubatch else 0.0
        return [len(document) / 1000 + drift for document in documents]


@dataclass(frozen=True)
class Bench:
    root: Path
    spec: ModelSpec
    candidates: Path
    compose_file: Path
    cpuinfo: Path

    def run(
        self, runner: RecordingRunner, client: ScoringClient, clock: FakeClock
    ) -> LocalRerankBenchmarkResult:
        manager = LlamaCppComposeManager(
            self.compose_file,
            runner=runner,
            client_factory=HealthyClient,
            environment={},
        )
        return run_local_rerank_benchmark(
            spec=self.spec,
            candidate_data_path=self.candidates,
            server=ComposeRerankServer(manager, self.spec, self.root),
            client_factory=lambda _endpoint: client,
            profile_root=self.root / "profiles",
            compose_file=self.compose_file,
            cpuinfo_path=self.cpuinfo,
            clock=clock,
        )


@pytest.fixture
def bench(tmp_path: Path) -> Bench:
    compose_file = tmp_path / "compose.yaml"
    compose_file.write_text(
        "services:\n  llama-reranker:\n"
        "    image: ghcr.io/ggml-org/llama.cpp:server-b10920\n",
        encoding="utf-8",
    )
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text(f"processor\t: 0\nmodel name\t: {CPU}\n", encoding="utf-8")
    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(b"gguf")
    spec = replace(
        require_model(MODEL),
        canonical_filename=artifact.name,
        byte_size=artifact.stat().st_size,
        sha256=file_sha256(artifact),
    )
    candidates = tmp_path / "candidates.jsonl"
    with candidates.open("w", encoding="utf-8") as handle:
        for query in range(8):
            row = {
                "query_id": f"q{query}",
                "query": f"query {query}",
                "candidates": [
                    {
                        "chunk_id": f"c{query}-{document}",
                        "document_text": "x" * (10 * query + document + 1),
                    }
                    for document in range(15)
                ],
            }
            handle.write(json.dumps(row) + "\n")
    return Bench(tmp_path, spec, candidates, compose_file, cpuinfo)


def test_benchmark_recreates_the_reranker_with_each_level_environment(bench: Bench):
    runner, clock = RecordingRunner(), FakeClock()

    bench.run(runner, ScoringClient(runner, clock), clock)

    assert [
        (
            up["LLAMA_RERANKER_PARALLEL"],
            up["LLAMA_RERANKER_UBATCH_SIZE"],
            up["LLAMA_RERANKER_CONTEXT_SIZE"],
            up["LLAMA_RERANKER_THREADS"],
        )
        for up in runner.ups
    ] == [
        ("4", "2048", "10240", "8"),
        ("4", "2048", "10240", "12"),
        ("8", "4096", "20480", "8"),
        ("8", "4096", "20480", "12"),
    ]


def test_each_level_sends_one_warm_up_and_six_full_groups(bench: Bench):
    runner, clock = RecordingRunner(), FakeClock()
    client = ScoringClient(runner, clock)

    bench.run(runner, client, clock)

    assert client.calls == [15] * (7 * 4)


def test_lowest_p95_level_becomes_the_local_profile(bench: Bench):
    runner, clock = RecordingRunner(), FakeClock()

    result = bench.run(runner, ScoringClient(runner, clock), clock)

    assert result.profile.selected == LOCAL_RERANK_SEARCH_SPACE.candidates[3]
    assert (
        result.path
        == bench.root / "profiles" / "rerank" / "qwen3_reranker_4b_fp16.json"
    )
    assert result.profile.identity.payload["machine_shape"] == CPU
    assert result.profile.sample_count == 90
    assert result.selected_measurement.latency_p95_seconds == pytest.approx(8 / 12)
    assert (
        load_local_rerank_profile(
            bench.spec,
            profile_root=bench.root / "profiles",
            compose_file=bench.compose_file,
            cpuinfo_path=bench.cpuinfo,
        )
        == result.profile
    )


def test_levels_whose_scores_drift_are_invalid(bench: Bench):
    runner, clock = RecordingRunner(), FakeClock()

    result = bench.run(
        runner, ScoringClient(runner, clock, drifting_ubatch="4096"), clock
    )

    assert [item.error_category for item in result.measurements] == [
        None,
        None,
        "score_mismatch",
        "score_mismatch",
    ]


def test_a_level_that_fails_to_start_keeps_the_service_logs(bench: Bench):
    runner = RecordingRunner(
        failing_ubatches=frozenset({"4096"}), logs="failed to allocate compute buffer"
    )
    clock = FakeClock()

    result = bench.run(runner, ScoringClient(runner, clock), clock)

    failed = result.measurements[2]
    assert (failed.status, failed.error_category) == ("invalid", "CalledProcessError")
    assert failed.log_tail is not None
    assert "failed to allocate compute buffer" in failed.log_tail
    assert result.profile.selected == LOCAL_RERANK_SEARCH_SPACE.candidates[1]


def test_every_level_failing_stops_with_the_log_tails(bench: Bench):
    runner = RecordingRunner(
        failing_ubatches=frozenset({"2048", "4096"}),
        logs="failed to allocate compute buffer",
    )
    clock = FakeClock()

    with pytest.raises(RuntimeError, match="failed to allocate compute buffer"):
        bench.run(runner, ScoringClient(runner, clock), clock)

    assert not (bench.root / "profiles").exists()


def test_cpu_model_name_reads_proc_cpuinfo_and_falls_back(tmp_path: Path):
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text(f"model name\t: {CPU}\n", encoding="utf-8")

    assert cpu_model_name(cpuinfo) == CPU
    assert cpu_model_name(tmp_path / "missing")
