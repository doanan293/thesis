from __future__ import annotations

import json
import subprocess
import threading
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GpuSample:
    gpu_index: int
    utilization_gpu: float
    utilization_memory: float
    memory_used_mib: float
    power_draw_watts: float
    temperature_celsius: float


@dataclass(frozen=True)
class OperationSample:
    server_index: int
    item_count: int
    input_characters: int
    latency_seconds: float
    status: str
    retries: int


def _number(value: str, label: str) -> float:
    try:
        result = float(value.strip())
    except ValueError as exc:
        raise ValueError(f"GPU telemetry row has invalid {label}") from exc
    if result < 0:
        raise ValueError(f"GPU telemetry row has negative {label}")
    return result


def parse_gpu_rows(text: str) -> list[GpuSample]:
    samples = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            raise ValueError("GPU telemetry row must contain six fields")
        try:
            gpu_index = int(fields[0])
        except ValueError as exc:
            raise ValueError("GPU telemetry row has invalid GPU index") from exc
        if gpu_index < 0:
            raise ValueError("GPU telemetry row has negative GPU index")
        samples.append(
            GpuSample(
                gpu_index,
                _number(fields[1], "GPU utilization"),
                _number(fields[2], "memory utilization"),
                _number(fields[3], "memory used"),
                _number(fields[4], "power draw"),
                _number(fields[5], "temperature"),
            )
        )
    return samples


class NvidiaSmiSampler:
    def __init__(self, interval_seconds: float = 2.0):
        self.interval_seconds = max(0.1, float(interval_seconds))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.status = "unavailable"
        self.error_category: str | None = None

    def start(self, callback: Callable[[GpuSample], None]) -> None:
        def run() -> None:
            while not self._stop.is_set():
                try:
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=5,
                    )
                    if result.returncode != 0:
                        self.error_category = "command_failed"
                    else:
                        for sample in parse_gpu_rows(result.stdout):
                            callback(sample)
                        self.status = "available"
                except FileNotFoundError:
                    self.error_category = "not_installed"
                    return
                except (OSError, subprocess.SubprocessError, ValueError):
                    self.error_category = "sampling_failed"
                self._stop.wait(self.interval_seconds)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=6)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


class RuntimeTelemetry:
    def __init__(
        self,
        stage: str,
        model: str,
        output_dir: Path,
        *,
        sample_interval_seconds: float = 2.0,
        sampler: NvidiaSmiSampler | None = None,
    ):
        self.stage = str(stage)
        self.model = str(model)
        self.output_dir = Path(output_dir)
        self.gpu_samples: list[GpuSample] = []
        self.operations: list[OperationSample] = []
        self.sampler = sampler or NvidiaSmiSampler(sample_interval_seconds)
        self.gpu_sampling_status = "unavailable"
        self.gpu_sampling_error: str | None = None
        self._closed = False
        try:
            self.sampler.start(self.record_gpu_sample)
        except (OSError, RuntimeError, ValueError) as exc:
            self.gpu_sampling_error = type(exc).__name__

    def record_gpu_sample(self, sample: GpuSample) -> None:
        self.gpu_samples.append(sample)
        self.gpu_sampling_status = "available"

    def record_operation(
        self,
        server_index: int,
        item_count: int,
        input_characters: int,
        latency_seconds: float,
        status: str,
        retries: int,
    ) -> None:
        if item_count < 0 or input_characters < 0 or latency_seconds < 0 or retries < 0:
            raise ValueError("telemetry operation values must be non-negative")
        self.operations.append(
            OperationSample(
                int(server_index),
                int(item_count),
                int(input_characters),
                float(latency_seconds),
                str(status),
                int(retries),
            )
        )

    def summary(self) -> dict:
        latency = [item.latency_seconds for item in self.operations]
        gpu_by_index: dict[int, list[GpuSample]] = defaultdict(list)
        for sample in self.gpu_samples:
            gpu_by_index[sample.gpu_index].append(sample)
        gpu = {}
        for index, samples in sorted(gpu_by_index.items()):
            gpu[str(index)] = {
                "utilization_gpu_mean": sum(s.utilization_gpu for s in samples)
                / len(samples),
                "utilization_gpu_p95": _percentile(
                    [s.utilization_gpu for s in samples], 0.95
                ),
                "utilization_memory_mean": sum(s.utilization_memory for s in samples)
                / len(samples),
                "memory_used_mib_peak": max(s.memory_used_mib for s in samples),
                "power_draw_watts_mean": sum(s.power_draw_watts for s in samples)
                / len(samples),
                "temperature_celsius_peak": max(
                    s.temperature_celsius for s in samples
                ),
                "sample_count": len(samples),
            }
        statuses: dict[str, int] = defaultdict(int)
        for operation in self.operations:
            statuses[operation.status] += 1
        return {
            "stage": self.stage,
            "model": self.model,
            "gpu_sampling_status": self.gpu_sampling_status,
            "gpu_sampling_error": self.gpu_sampling_error
            or getattr(self.sampler, "error_category", None),
            "operations": {
                "count": len(self.operations),
                "items": sum(item.item_count for item in self.operations),
                "input_characters": sum(
                    item.input_characters for item in self.operations
                ),
                "retries": sum(item.retries for item in self.operations),
                "latency_p50_seconds": _percentile(latency, 0.50),
                "latency_p95_seconds": _percentile(latency, 0.95),
                "statuses": dict(sorted(statuses.items())),
            },
            "gpu": gpu,
        }

    def write_report(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        target = self.output_dir / "telemetry.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.summary(), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
        return target

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.sampler.stop()
