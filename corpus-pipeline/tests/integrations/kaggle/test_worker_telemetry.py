import json
from pathlib import Path

import pytest

from corpus_pipeline.integrations.kaggle.workers.telemetry import (
    GpuSample,
    RuntimeTelemetry,
    parse_gpu_rows,
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
    )
    telemetry.record_operation(0, 1, 321, 1.2, "success", 0)
    telemetry.record_operation(0, 1, 500, 2.0, "retry_success", 1)

    path = telemetry.write_report()
    report = json.loads(path.read_text(encoding="utf-8"))

    assert report["gpu_sampling_status"] == "available"
    assert report["operations"]["count"] == 2
    assert report["operations"]["retries"] == 1
    assert report["operations"]["latency_p95_seconds"] == pytest.approx(1.96)
    assert report["gpu"]["0"]["utilization_gpu_mean"] == pytest.approx(85.0)
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
