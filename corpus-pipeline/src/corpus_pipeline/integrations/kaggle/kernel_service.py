from __future__ import annotations

import random
import time
from pathlib import Path

from corpus_pipeline.integrations.kaggle.api import (
    KaggleCommandRunner,
    kernel_logs_command,
    kernel_output_command,
    kernel_push_command,
    kernel_status_command,
)
from corpus_pipeline.integrations.kaggle.errors import (
    ErrorDisposition,
    KaggleCommandError,
    classify_command_error,
)
from corpus_pipeline.integrations.kaggle.log_follower import (
    ManagedLogFollower,
    retry_delay_seconds,
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
        jitter=random.random,
        emit=print,
        follower_factory=ManagedLogFollower,
    ):
        self.runner = runner
        self.owner = owner
        self.poll_interval_seconds = max(0.1, float(poll_interval_seconds))
        self._sleep = sleep
        self._monotonic = monotonic
        self._jitter = jitter
        self._emit = emit
        self._follower_factory = follower_factory

    def push(self, bundle: Path, *, timeout_seconds: int = 43_200) -> None:
        self.runner.run(kernel_push_command(bundle, timeout_seconds))

    def download_output(self, reference: str, destination: Path) -> None:
        self.runner.run(kernel_output_command(reference, destination))

    def poll(self, reference: str, *, timeout_seconds: float = 43_200) -> None:
        if self.runner.dry_run:
            self.runner.run(kernel_status_command(reference), capture_output=True)
            return
        started = self._monotonic()
        follower = self._follower_factory(self.runner, reference)
        retry_at = started
        consecutive_failures = 0
        degradation_reported = False
        last_status = None
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
                    follower.stop()
                    raise RuntimeError(
                        f"Kernel {reference} failed: {self._log_tail(reference)}"
                    )

                if status != last_status:
                    self._emit(
                        f"[{format_elapsed(elapsed)}] Kaggle kernel {status.lower()}"
                    )
                    last_status = status

                failure = follower.poll_failure()
                if failure is not None:
                    if failure.disposition is ErrorDisposition.FATAL:
                        follower.stop()
                        raise RuntimeError(
                            f"Kaggle live log stream failed: {failure.detail}"
                        )
                    consecutive_failures += 1
                    retry_at = self._monotonic() + retry_delay_seconds(
                        consecutive_failures,
                        jitter=self._jitter(),
                        retry_after_seconds=failure.retry_after_seconds,
                    )
                    if consecutive_failures >= 2 and not degradation_reported:
                        self._emit(
                            "Kaggle live logs temporarily unavailable; "
                            "kernel status monitoring continues."
                        )
                        degradation_reported = True

                if (
                    status == "RUNNING"
                    and not follower.is_running
                    and self._monotonic() >= retry_at
                ):
                    try:
                        follower.start()
                    except OSError:
                        consecutive_failures += 1
                        retry_at = self._monotonic() + retry_delay_seconds(
                            consecutive_failures, jitter=self._jitter()
                        )
                        if consecutive_failures >= 2 and not degradation_reported:
                            self._emit(
                                "Kaggle live logs temporarily unavailable; "
                                "kernel status monitoring continues."
                            )
                            degradation_reported = True
                self._sleep(self.poll_interval_seconds)
        finally:
            follower.stop()

    def _log_tail(self, reference: str) -> str:
        try:
            entries = parse_kernel_log_entries(
                self.runner.run(kernel_logs_command(reference), capture_output=True)
            )
        except Exception as error:
            return f"log unavailable: {error}"
        return "\n".join(entries[-3:])
