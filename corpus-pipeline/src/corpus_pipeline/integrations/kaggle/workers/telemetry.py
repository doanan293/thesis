from __future__ import annotations

import json
import math
import subprocess
import threading
import time
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
    prompt_tokens_cached: int | None = None
    prompt_tokens_evaluated: int | None = None


@dataclass(frozen=True)
class ProcessRssSample:
    server_index: int
    rss_mib: float
    elapsed_seconds: float = 0.0


def parse_process_rss_mib(text: str) -> float:
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            fields = line.split()
            if len(fields) < 2:
                break
            value = float(fields[1])
            if value < 0:
                raise ValueError("process RSS must be non-negative")
            return value / 1024.0
    raise ValueError("VmRSS is missing")


class ProcessRssSampler:
    def __init__(self, interval_seconds: float = 2.0):
        self.interval_seconds = max(0.1, float(interval_seconds))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._processes: dict[int, int] = {}
        self._lock = threading.Lock()
        self.status = "unavailable"
        self.error_category: str | None = None

    def register(self, processes: dict[int, int]) -> None:
        with self._lock:
            self._processes = dict(processes)

    def start(self, callback: Callable[[ProcessRssSample], None]) -> None:
        started = time.monotonic()

        def run() -> None:
            while not self._stop.is_set():
                with self._lock:
                    processes = dict(self._processes)
                for server_index, pid in processes.items():
                    try:
                        sample = parse_process_rss_mib(
                            Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
                        )
                        callback(
                            ProcessRssSample(
                                server_index,
                                sample,
                                max(0.0, time.monotonic() - started),
                            )
                        )
                        self.status = "available"
                    except (OSError, ValueError):
                        self.error_category = "sampling_failed"
                self._stop.wait(self.interval_seconds)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=6)


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
        process_sampler: ProcessRssSampler | None = None,
    ):
        self.stage = str(stage)
        self.model = str(model)
        self.output_dir = Path(output_dir)
        self.gpu_samples: list[GpuSample] = []
        self.operations: list[OperationSample] = []
        self.sampler = sampler or NvidiaSmiSampler(sample_interval_seconds)
        self.process_sampler = process_sampler or ProcessRssSampler(
            sample_interval_seconds
        )
        self.process_rss_samples: list[ProcessRssSample] = []
        self.gpu_sampling_status = "unavailable"
        self.gpu_sampling_error: str | None = None
        self._closed = False
        try:
            self.sampler.start(self.record_gpu_sample)
        except (OSError, RuntimeError, ValueError) as exc:
            self.gpu_sampling_error = type(exc).__name__
        try:
            self.process_sampler.start(self.record_process_rss_sample)
        except (OSError, RuntimeError, ValueError) as exc:
            self.process_sampling_error = type(exc).__name__
        else:
            self.process_sampling_error = None

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
        prompt_tokens_cached: int | None = None,
        prompt_tokens_evaluated: int | None = None,
    ) -> None:
        if item_count < 0 or input_characters < 0 or latency_seconds < 0 or retries < 0:
            raise ValueError("telemetry operation values must be non-negative")
        for value in (prompt_tokens_cached, prompt_tokens_evaluated):
            if value is not None and (isinstance(value, bool) or value < 0):
                raise ValueError("prompt timing values must be non-negative integers")
        self.operations.append(
            OperationSample(
                int(server_index),
                int(item_count),
                int(input_characters),
                float(latency_seconds),
                str(status),
                int(retries),
                prompt_tokens_cached,
                prompt_tokens_evaluated,
            )
        )

    def register_server_processes(self, processes: dict[int, int]) -> None:
        self.process_sampler.register(processes)

    def record_process_rss_sample(self, sample: ProcessRssSample) -> None:
        self.process_rss_samples.append(sample)

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
                "temperature_celsius_peak": max(s.temperature_celsius for s in samples),
                "sample_count": len(samples),
            }
        statuses: dict[str, int] = defaultdict(int)
        for operation in self.operations:
            statuses[operation.status] += 1
        cached_values = [
            item.prompt_tokens_cached
            for item in self.operations
            if item.prompt_tokens_cached is not None
        ]
        evaluated_values = [
            item.prompt_tokens_evaluated
            for item in self.operations
            if item.prompt_tokens_evaluated is not None
        ]
        cached_total = sum(cached_values) if cached_values else None
        evaluated_total = sum(evaluated_values) if evaluated_values else None
        timing_total = (
            cached_total + evaluated_total
            if cached_total is not None and evaluated_total is not None
            else None
        )
        rss_by_server: dict[int, list[ProcessRssSample]] = defaultdict(list)
        for sample in self.process_rss_samples:
            rss_by_server[sample.server_index].append(sample)
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
                "prompt_tokens_cached": cached_total,
                "prompt_tokens_evaluated": evaluated_total,
                "prompt_cache_hit_ratio": (
                    cached_total / timing_total if timing_total else None
                ),
                "statuses": dict(sorted(statuses.items())),
            },
            "gpu": gpu,
            "process_sampling_status": getattr(
                self.process_sampler, "status", "unavailable"
            ),
            "process_sampling_error": getattr(self, "process_sampling_error", None)
            or getattr(self.process_sampler, "error_category", None),
            "servers": {
                str(index): {
                    "rss_mib_start": values[0].rss_mib,
                    "rss_mib_peak": max(item.rss_mib for item in values),
                    "rss_mib_final": values[-1].rss_mib,
                    "post_warmup_slope_mib_per_minute": self._rss_slope(values),
                    "sample_count": len(values),
                }
                for index, values in sorted(rss_by_server.items())
            },
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
        self.process_sampler.stop()

    @staticmethod
    def _rss_slope(samples: list[ProcessRssSample]) -> float | None:
        if len(samples) < 2:
            return None
        first_time = samples[0].elapsed_seconds
        last_time = samples[-1].elapsed_seconds
        warmup_cutoff = first_time + (last_time - first_time) * 0.2
        post_warmup = [
            item for item in samples if item.elapsed_seconds >= warmup_cutoff
        ]
        if len(post_warmup) < 2:
            return None
        start, end = post_warmup[0], post_warmup[-1]
        elapsed_minutes = (end.elapsed_seconds - start.elapsed_seconds) / 60.0
        if elapsed_minutes <= 0:
            return None
        slope = (end.rss_mib - start.rss_mib) / elapsed_minutes
        return slope if math.isfinite(slope) else None
