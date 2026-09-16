from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Protocol

from pharma_lab.integrations.kaggle.api import (
    kernel_list_mine_command,
    kernel_logs_command,
    kernel_output_command,
    kernel_push_command,
    kernel_status_command,
)
from pharma_lab.integrations.kaggle.errors import (
    ErrorDisposition,
    KaggleCommandError,
    KaggleDetached,
    KaggleOutputUnavailable,
    classify_command_error,
)
from pharma_lab.integrations.kaggle.log_follower import (
    FollowerCommandRunner,
    ManagedLogFollower,
    retry_delay_seconds,
)
from pharma_lab.integrations.kaggle.models import (
    KernelPresence,
    KernelRemoteState,
    KernelStatus,
)
from pharma_lab.integrations.kaggle.parsers import (
    format_elapsed,
    parse_kernel_log_entries,
    parse_kernel_references,
    parse_kernel_status,
)


class KernelCommandRunner(FollowerCommandRunner, Protocol):
    """Kaggle CLI runner used by KernelService."""

    @property
    def dry_run(self) -> bool: ...

    def run(self, args: list[str], /, capture_output: bool = False) -> str: ...


class KernelService:
    def __init__(
        self,
        runner: KernelCommandRunner,
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

    def inspect_state(self, reference: str) -> KernelRemoteState:
        try:
            output = self.runner.run(
                kernel_status_command(reference), capture_output=True
            )
        except KaggleCommandError as error:
            detail = f"{error.stdout}\n{error.stderr}".strip()
            folded = detail.casefold()
            if "404" in folded or "not found" in folded:
                return KernelRemoteState(
                    reference, KernelPresence.ABSENT, detail=detail
                )
            return KernelRemoteState(reference, KernelPresence.UNKNOWN, detail=detail)
        try:
            status = parse_kernel_status(output)
        except (RuntimeError, ValueError) as error:
            return KernelRemoteState(
                reference, KernelPresence.UNKNOWN, detail=str(error)
            )
        return KernelRemoteState(reference, KernelPresence.EXISTS, status=status)

    def push(self, bundle: Path, *, timeout_seconds: int = 43_200) -> None:
        self.runner.run(kernel_push_command(bundle, timeout_seconds))

    def confirm_missing(self, reference: str) -> bool | None:
        """Return whether an inaccessible reference is absent from owned kernels."""
        try:
            output = self.runner.run(
                kernel_list_mine_command(reference.rsplit("/", 1)[-1]),
                capture_output=True,
            )
        except KaggleCommandError:
            return None
        return reference not in parse_kernel_references(output)

    def download_output(self, reference: str, destination: Path) -> None:
        for attempt in range(1, 6):
            try:
                self.runner.run(kernel_output_command(reference, destination))
                return
            except KaggleCommandError as error:
                detail = f"{error.stdout}\n{error.stderr}".strip()
                folded = detail.casefold()
                if "404" in folded or "not found" in folded or "no output" in folded:
                    raise KaggleOutputUnavailable(
                        f"Kernel output is unavailable for {reference}: {detail[:500]}"
                    ) from error
                if classify_command_error(error) is ErrorDisposition.FATAL:
                    raise
                if attempt == 5:
                    raise
                self._sleep(retry_delay_seconds(attempt, jitter=self._jitter()))

    def wait_for_terminal(
        self, reference: str, *, timeout_seconds: float = 43_200
    ) -> KernelStatus:
        if self.runner.dry_run:
            self.runner.run(kernel_status_command(reference), capture_output=True)
            return KernelStatus.COMPLETE
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
                if status is KernelStatus.COMPLETE:
                    return status
                if status is KernelStatus.ERROR:
                    return status

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
                    status is KernelStatus.RUNNING
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
        except KeyboardInterrupt as error:
            raise KaggleDetached(reference) from error
        finally:
            follower.stop()

    def poll(self, reference: str, *, timeout_seconds: float = 43_200) -> None:
        status = self.wait_for_terminal(reference, timeout_seconds=timeout_seconds)
        if status is KernelStatus.ERROR:
            raise RuntimeError(f"Kernel {reference} failed: {self.log_tail(reference)}")

    def log_tail(self, reference: str) -> str:
        try:
            entries = parse_kernel_log_entries(
                self.runner.run(kernel_logs_command(reference), capture_output=True)
            )
        except Exception as error:
            return f"log unavailable: {error}"
        return "\n".join(entries[-3:])
