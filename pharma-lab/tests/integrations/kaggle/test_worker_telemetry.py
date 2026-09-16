import json
from pathlib import Path

import pytest

from pharma_lab.integrations.kaggle.workers.telemetry import (
    GpuSample,
    ProcessRssSample,
    RuntimeTelemetry,
    parse_gpu_rows,
    parse_process_rss_mib,
)


class FakeSampler:
    def __init__(self, samples: list[GpuSample] | None = None, error=None):
        self.samples = samples or []
        self.error = error
        self.callback = None

    def start(self, callback):
        self.callback = callback
        if self.error is not None:
            raise self.error
        for sample in self.samples:
            callback(sample)

    def stop(self):
        return None


class FakeProcessSampler:
    status = "available"
    error_category = None

    def start(self, callback):
        callback(ProcessRssSample(0, 100.0, 0.0))
        callback(ProcessRssSample(0, 110.0, 60.0))
        callback(ProcessRssSample(0, 112.0, 120.0))

    def register(self, _processes):
        return None

    def stop(self):
        return None


def test_gpu_sampler_parses_nvidia_smi_rows():
    samples = parse_gpu_rows("0, 91, 12, 8123, 64.5, 71\n1, 88, 9, 7900, 62.0, 69\n")

    assert samples == [
        GpuSample(0, 91.0, 12.0, 8123.0, 64.5, 71.0),
        GpuSample(1, 88.0, 9.0, 7900.0, 62.0, 69.0),
    ]


def test_telemetry_report_aggregates_operations_and_gpu_samples(tmp_path: Path):
    telemetry = RuntimeTelemetry(
        "rerank",
        "qwen3-reranker:0.6b-fp16",
        tmp_path,
        sampler=FakeSampler(
            [GpuSample(0, 90, 10, 8000, 60, 70), GpuSample(0, 80, 10, 8100, 62, 72)]
        ),
        process_sampler=FakeProcessSampler(),
    )
    telemetry.record_operation(0, 1, 321, 1.2, "success", 0)
    telemetry.record_operation(0, 1, 500, 2.0, "retry_success", 1)

    path = telemetry.write_report()
    report = json.loads(path.read_text(encoding="utf-8"))

    assert report["gpu_sampling_status"] == "available"
    assert report["operations"]["count"] == 2
    assert report["operations"]["retries"] == 1
    assert report["operations"]["latency_p95_seconds"] == pytest.approx(1.96)
    assert "prompt_cache_hit_ratio" not in report["operations"]
    assert report["gpu"]["0"]["utilization_gpu_mean"] == pytest.approx(85.0)
    assert report["servers"]["0"]["rss_mib_start"] == pytest.approx(100.0)
    assert report["servers"]["0"]["rss_mib_peak"] == pytest.approx(112.0)
    assert report["servers"]["0"]["rss_mib_final"] == pytest.approx(112.0)
    assert report["servers"]["0"]["post_warmup_slope_mib_per_minute"] == pytest.approx(
        2.0
    )
    assert "query secret" not in path.read_text(encoding="utf-8")


def test_missing_nvidia_smi_is_non_fatal(tmp_path: Path):
    telemetry = RuntimeTelemetry(
        "rerank",
        "qwen3-reranker:0.6b-fp16",
        tmp_path,
        sampler=FakeSampler(error=FileNotFoundError("nvidia-smi")),
    )

    assert telemetry.summary()["gpu_sampling_status"] == "unavailable"


def test_invalid_gpu_row_is_rejected():
    with pytest.raises(ValueError, match="GPU telemetry row"):
        parse_gpu_rows("0, 91, N/A, 8123, 64.5\n")


def test_process_rss_parser_converts_kib_to_mib():
    assert parse_process_rss_mib(
        "Name:\tllama-server\nVmRSS:\t2048 kB\n"
    ) == pytest.approx(2.0)
