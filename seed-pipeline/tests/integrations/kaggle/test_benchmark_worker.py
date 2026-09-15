import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import seed_pipeline.integrations.kaggle.workers.benchmark as benchmark_worker
from seed_pipeline.integrations.kaggle.artifacts import sha256_file
from seed_pipeline.integrations.kaggle.workers.benchmark import run_benchmark_worker
from seed_pipeline.runtime.benchmarking import BenchmarkMeasurement, LevelResult
from seed_pipeline.runtime.catalog import require_model
from seed_pipeline.runtime.runtime_profiles import RuntimeCandidate

MODEL = "qwen3-reranker:0.6b-fp16"


def _levels() -> tuple[RuntimeCandidate, ...]:
    space = require_model(MODEL).rerank_search_space
    assert space is not None
    return space.candidates


def _config(tmp_path: Path, **extra: object) -> dict:
    return {
        "output_dir": str(tmp_path / "output"),
        "model": MODEL,
        "stage": "rerank-benchmark",
        "benchmark_items": 960,
        "benchmark_groups": 32,
        "benchmark_levels": [level.to_dict() for level in _levels()],
        "identity": {"stage": "rerank-benchmark"},
        "job_sha256": "a" * 64,
        **extra,
    }


def _ok(
    candidate: RuntimeCandidate, seconds: float, scores: dict[str, float]
) -> LevelResult:
    return LevelResult(
        BenchmarkMeasurement(
            candidate,
            960,
            100_000,
            seconds,
            latency_p50_seconds=0.5,
            latency_p95_seconds=1.0,
        ),
        scores,
    )


def _rows(config: dict) -> list[dict]:
    path = Path(config["output_dir"]) / "benchmark_results.jsonl"
    return [
        json.loads(line)["measurement"]
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_benchmark_worker_keeps_failed_levels_with_their_server_log(tmp_path):
    config = _config(tmp_path)
    levels = _levels()

    def measure(index, candidate):
        if index == 1:
            log = Path(config["output_dir"]) / "level-1" / "server-0.log"
            log.parent.mkdir(parents=True)
            log.write_text("ggml_cuda: out of memory\n", encoding="utf-8")
            raise RuntimeError("replica(s) 0 exited")
        return _ok(candidate, 10.0 if index == 0 else 8.0, {"q\x1fc": 0.5})

    artifact = run_benchmark_worker(config, measure_level=measure)

    rows = _rows(config)
    assert [row["status"] for row in rows] == ["ok", "invalid", "ok"]
    assert rows[1]["error_category"] == "RuntimeError"
    assert "out of memory" in rows[1]["log_tail"]
    assert [row["candidate"] for row in rows] == [level.to_dict() for level in levels]
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    assert manifest["runtime"] == {
        "recommendation": levels[2].to_dict(),
        "sample_count": 960,
    }
    assert (Path(config["output_dir"]) / "benchmark_report.md").is_file()


def test_benchmark_worker_rejects_a_level_whose_scores_drift(tmp_path):
    config = _config(tmp_path)
    levels = _levels()

    def measure(index, candidate):
        drift = 0.01 if index == 2 else 0.0
        return _ok(candidate, 10.0 - index, {"q\x1fc": 0.5 + drift})

    artifact = run_benchmark_worker(config, measure_level=measure)

    rows = _rows(config)
    assert [row["status"] for row in rows] == ["ok", "ok", "invalid"]
    assert rows[2]["error_category"] == "score_mismatch"
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    assert manifest["runtime"]["recommendation"] == levels[1].to_dict()


def test_benchmark_worker_records_no_recommendation_when_every_level_fails(tmp_path):
    config = _config(tmp_path)

    def measure(_index, _candidate):
        raise RuntimeError("server did not start")

    artifact = run_benchmark_worker(config, measure_level=measure)

    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    assert manifest["runtime"]["recommendation"] is None
    assert artifact.completion.complete == len(_levels())


def _write_candidates(path: Path, queries: int, documents: int) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for query in range(queries):
            row = {
                "query_id": f"q{query}",
                "query": f"query {query}",
                "candidates": [
                    {
                        "chunk_id": f"c{query}-{document}",
                        "document_text": "x" * (query + document + 1),
                    }
                    for document in range(documents)
                ],
            }
            handle.write(json.dumps(row) + "\n")


def test_rerank_level_restarts_servers_and_sends_warm_up_then_full_groups(
    tmp_path, monkeypatch
):
    candidates = tmp_path / "candidates.jsonl"
    _write_candidates(candidates, queries=40, documents=30)
    config = _config(
        tmp_path,
        input_files={
            "candidates": {
                "path": str(candidates),
                "filename": candidates.name,
                "sha256": sha256_file(candidates),
            }
        },
    )
    started_levels: list[dict] = []
    calls: list[tuple[str, int]] = []

    @contextmanager
    def fake_servers(level_config, *, telemetry=None):
        del telemetry
        started_levels.append(level_config)
        yield [
            SimpleNamespace(base_url="http://127.0.0.1:11434"),
            SimpleNamespace(base_url="http://127.0.0.1:11435"),
        ]

    class FakeClient:
        def __init__(self, base_url):
            self.base_url = base_url

        def rerank_native(self, query, documents, model):
            assert model == MODEL
            calls.append((query, len(documents)))
            return [len(document) / 100 for document in documents]

    monkeypatch.setattr(benchmark_worker, "managed_model_servers", fake_servers)
    monkeypatch.setattr(benchmark_worker, "LlamaCppClient", FakeClient)

    artifact = run_benchmark_worker(config)

    levels = _levels()
    assert [item["runtime_overrides"] for item in started_levels] == [
        level.to_dict() for level in levels
    ]
    assert [Path(item["output_dir"]).name for item in started_levels] == [
        "level-0",
        "level-1",
        "level-2",
    ]
    # One untimed warm-up group, then 32 timed groups of 30 documents, per level.
    assert len(calls) == len(levels) * 33
    assert {size for _query, size in calls} == {30}
    rows = _rows(config)
    assert all(row["status"] == "ok" and row["items"] == 960 for row in rows)
    assert all(row["max_abs_score_delta"] == 0.0 for row in rows)
    assert artifact.completion.complete == 3


def test_benchmark_worker_main_loads_config_and_invokes_worker(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config = {"output_dir": str(tmp_path), "model": "model"}
    config_path.write_text(json.dumps(config), encoding="utf-8")
    captured = {}

    def fake_worker(received):
        captured["config"] = received

    monkeypatch.setenv("KAGGLE_PIPELINE_CONFIG", str(config_path))
    monkeypatch.setattr(benchmark_worker, "run_benchmark_worker", fake_worker)

    assert benchmark_worker.main() == 0
    assert captured["config"] == config
