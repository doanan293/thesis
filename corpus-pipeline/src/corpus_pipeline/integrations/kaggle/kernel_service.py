from __future__ import annotations

import time
from pathlib import Path

from corpus_pipeline.integrations.kaggle.api import (
    KaggleCommandRunner,
    kernel_logs_command,
    kernel_logs_follow_command,
    kernel_output_command,
    kernel_push_command,
    kernel_status_command,
)
from corpus_pipeline.integrations.kaggle.errors import (
    ErrorDisposition,
    KaggleCommandError,
    classify_command_error,
)
from corpus_pipeline.integrations.kaggle.parsers import (
    format_elapsed,
    parse_kernel_log_entries,
    parse_kernel_status,
)


class KernelService:
    def __init__(
        self,
        runner: KaggleCommandRunner,
        owner: str,
        *,
        poll_interval_seconds: float = 15,
        sleep=time.sleep,
        monotonic=time.monotonic,
    ):
        self.runner = runner
        self.owner = owner
        self.poll_interval_seconds = max(0.1, float(poll_interval_seconds))
        self._sleep = sleep
        self._monotonic = monotonic

    def push(self, bundle: Path, *, timeout_seconds: int = 43_200) -> None:
        self.runner.run(kernel_push_command(bundle, timeout_seconds))

    def download_output(self, reference: str, destination: Path) -> None:
        self.runner.run(kernel_output_command(reference, destination))

    def poll(self, reference: str, *, timeout_seconds: float = 43_200) -> None:
        if self.runner.dry_run:
            self.runner.run(kernel_status_command(reference), capture_output=True)
            return
        log_process = self.runner.start(kernel_logs_follow_command(reference))
        started = self._monotonic()
        try:
            while True:
                elapsed = self._monotonic() - started
                if elapsed >= timeout_seconds:
                    raise TimeoutError(
                        f"Kernel {reference} timed out after {format_elapsed(elapsed)}"
                    )
                try:
                    status = parse_kernel_status(
                        self.runner.run(
                            kernel_status_command(reference), capture_output=True
                        )
                    )
                except KaggleCommandError as error:
                    if classify_command_error(error) is ErrorDisposition.FATAL:
                        raise
                    self._sleep(self.poll_interval_seconds)
                    continue
                if status == "COMPLETE":
                    return
                if status == "ERROR":
                    raise RuntimeError(
                        f"Kernel {reference} failed: {self._log_tail(reference)}"
                    )
                self._sleep(self.poll_interval_seconds)
        finally:
            if log_process is not None:
                log_process.terminate()

    def _log_tail(self, reference: str) -> str:
        try:
            entries = parse_kernel_log_entries(
                self.runner.run(kernel_logs_command(reference), capture_output=True)
            )
        except Exception as error:
            return f"log unavailable: {error}"
        return "\n".join(entries[-3:])
