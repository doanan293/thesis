import json

import corpus_pipeline.integrations.kaggle.workers.benchmark as benchmark_worker
from corpus_pipeline.integrations.kaggle.workers.benchmark import run_benchmark_worker
from corpus_pipeline.runtime.benchmarking import BenchmarkMeasurement


def test_benchmark_worker_attempts_all_levels_and_keeps_invalid_level(tmp_path):
    config = {
        "output_dir": str(tmp_path),
        "model": "qwen3-reranker:0.6b-fp16",
        "stage": "rerank-benchmark",
        "benchmark_items": 4,
        "benchmark_levels": [
            {"batch_size": 1, "concurrency": 1},
            {"batch_size": 1, "concurrency": 2},
        ],
        "identity": {"stage": "rerank-benchmark"},
        "job_sha256": "a" * 64,
    }

    def measure(level):
        if level.concurrency == 2:
            raise RuntimeError("out of memory")
        return BenchmarkMeasurement(level, 4, 100, 1.0)

    artifact = run_benchmark_worker(config, measure_level=measure)

    assert artifact.completion.complete == 2
    rows = [
        json.loads(line)
        for line in (tmp_path / "benchmark_results.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2
    assert rows[1]["measurement"]["status"] == "invalid"
    assert (tmp_path / "benchmark_report.md").is_file()


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
