from __future__ import annotations

import re
import subprocess
import threading
from dataclasses import dataclass
from typing import Protocol, TextIO

from corpus_pipeline.integrations.kaggle.api import kernel_logs_follow_command
from corpus_pipeline.integrations.kaggle.errors import ErrorDisposition

_TRANSIENT_MARKERS = (
    "408",
    "429",
    "500",
    "502",
    "503",
    "504",
    "connection",
    "timed out",
    "timeout",
    "response ended prematurely",
)
_FATAL_MARKERS = ("401", "403", "permission", "owner mismatch")
_RETRY_AFTER = re.compile(r"retry-after\s*[:=]\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


@dataclass(frozen=True)
class FollowerFailure:
    disposition: ErrorDisposition
    detail: str
    retry_after_seconds: float | None = None


def classify_follower_stderr(stderr: str) -> FollowerFailure:
    detail = stderr.strip() or "Kaggle log follower exited without an error message"
    folded = detail.casefold()
    retry_after_match = _RETRY_AFTER.search(detail)
    retry_after = (
        float(retry_after_match.group(1)) if retry_after_match is not None else None
    )
    if any(marker in folded for marker in _FATAL_MARKERS):
        disposition = ErrorDisposition.FATAL
    elif any(marker in folded for marker in _TRANSIENT_MARKERS):
        disposition = ErrorDisposition.RETRYABLE
    else:
        disposition = ErrorDisposition.RETRYABLE
    return FollowerFailure(disposition, detail[:2000], retry_after)


def retry_delay_seconds(
    failure_count: int,
    *,
    jitter: float,
    retry_after_seconds: float | None = None,
) -> float:
    exponential = min(60.0, 2.0 ** max(0, failure_count - 1))
    delayed = max(exponential, retry_after_seconds or 0.0)
    return min(60.0, delayed + max(0.0, jitter))


def _emit_stdout(line: str) -> None:
    print(line, end="", flush=True)


class FollowerCommandRunner(Protocol):
    """Kaggle CLI runner used to stream kernel logs."""

    def start(
        self, args: list[str], /, *, capture_output: bool = False
    ) -> subprocess.Popen[str] | None: ...

    def redact(self, value: str, /) -> str: ...


class ManagedLogFollower:
    def __init__(
        self, runner: FollowerCommandRunner, reference: str, *, emit=_emit_stdout
    ):
        self.runner = runner
        self.reference = reference
        self.emit = emit
        self.process: subprocess.Popen[str] | None = None
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr_parts: list[str] = []
        self._history: list[str] = []
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self) -> None:
        process = self.runner.start(
            kernel_logs_follow_command(self.reference), capture_output=True
        )
        if process is None or process.stdout is None or process.stderr is None:
            raise OSError("Kaggle log follower did not expose captured streams")
        self.process = process
        self._stderr_parts = []
        self._stdout_thread = threading.Thread(
            target=self._pump_stdout, args=(process.stdout,), daemon=True
        )
        self._stderr_thread = threading.Thread(
            target=self._pump_stderr, args=(process.stderr,), daemon=True
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

    def _pump_stdout(self, stream: TextIO) -> None:
        for index, line in enumerate(stream):
            with self._lock:
                if index < len(self._history) and self._history[index] == line:
                    continue
                if index < len(self._history):
                    self._history[index:] = [line]
                else:
                    self._history.append(line)
            self.emit(line)

    def _pump_stderr(self, stream: TextIO) -> None:
        self._stderr_parts.extend(stream)

    def wait_for_output_for_test(self) -> None:
        if self._stdout_thread is not None:
            self._stdout_thread.join(timeout=1)

    def poll_failure(self) -> FollowerFailure | None:
        if self.process is None or self.process.poll() is None:
            return None
        self._join_readers()
        returncode = self.process.returncode
        self.process = None
        if returncode == 0:
            return None
        return classify_follower_stderr(self.runner.redact("".join(self._stderr_parts)))

    def stop(self) -> None:
        process = self.process
        self.process = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        self._join_readers()

    def _join_readers(self) -> None:
        for thread in (self._stdout_thread, self._stderr_thread):
            if thread is not None:
                thread.join(timeout=1)
