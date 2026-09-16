from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from pharma_lab.integrations.kaggle.errors import (
    KaggleDetached,
    KaggleOutputUnavailable,
    KaggleRemoteStateError,
)
from pharma_lab.integrations.kaggle.models import (
    ActionVerb,
    KernelPresence,
    KernelRemoteState,
    KernelStatus,
    ReconcileAction,
    StageJob,
)


@dataclass(frozen=True)
class KernelResolution:
    remote: KernelRemoteState
    output_root: Path | None
    actions: tuple[ReconcileAction, ...]
    submitted: bool = False


class ReconcilerKernels(Protocol):
    """Pipeline kernel operations KernelReconciler depends on."""

    def reference(self, job: StageJob, /) -> str: ...

    def discover(self, job: StageJob, /) -> KernelRemoteState: ...

    def push(self, bundle: Path, /, *, timeout_seconds: int) -> None: ...

    def wait_for_terminal(
        self, reference: str, /, *, timeout_seconds: int
    ) -> KernelStatus: ...

    def download_output(self, reference: str, destination: Path, /) -> None: ...

    def log_tail(self, reference: str, /) -> str: ...


class KernelReconciler:
    def __init__(self, kernels: ReconcilerKernels):
        self.kernels = kernels

    def reference(self, job: StageJob) -> str:
        return self.kernels.reference(job)

    def log_tail(self, reference: str) -> str:
        return self.kernels.log_tail(reference)

    def inspect(self, job: StageJob) -> KernelRemoteState:
        return self.kernels.discover(job)

    def attach_or_recover(
        self,
        job: StageJob,
        remote: KernelRemoteState,
        staging_root: Path,
        *,
        timeout_seconds: int,
    ) -> KernelResolution:
        del job
        status = remote.status
        actions: list[ReconcileAction] = []
        if status in {KernelStatus.QUEUED, KernelStatus.RUNNING}:
            actions.append(
                ReconcileAction(
                    "kernel", remote.reference, ActionVerb.ATTACH, status.value
                )
            )
            status = self.kernels.wait_for_terminal(
                remote.reference, timeout_seconds=timeout_seconds
            )
            remote = replace(remote, status=status)
        if status not in {KernelStatus.COMPLETE, KernelStatus.ERROR}:
            raise KaggleRemoteStateError(
                f"Kernel {remote.reference} has no terminal status"
            )
        destination = Path(staging_root) / "downloaded-output"
        try:
            self.kernels.download_output(remote.reference, destination)
        except KeyboardInterrupt as error:
            raise KaggleDetached(remote.reference) from error
        except KaggleOutputUnavailable:
            if status is KernelStatus.ERROR:
                return KernelResolution(remote, None, tuple(actions))
            raise
        actions.append(
            ReconcileAction(
                "artifact", remote.reference, ActionVerb.DOWNLOAD, status.value
            )
        )
        return KernelResolution(remote, destination, tuple(actions))

    def submit(
        self,
        job: StageJob,
        bundle: Path,
        staging_root: Path,
        *,
        timeout_seconds: int,
    ) -> KernelResolution:
        reference = self.kernels.reference(job)
        try:
            self.kernels.push(bundle, timeout_seconds=timeout_seconds)
        except KeyboardInterrupt as error:
            raise KaggleDetached(reference) from error
        queued = KernelRemoteState(
            reference, KernelPresence.EXISTS, KernelStatus.QUEUED
        )
        result = self.attach_or_recover(
            job, queued, staging_root, timeout_seconds=timeout_seconds
        )
        submitted = ReconcileAction(
            "kernel", reference, ActionVerb.SUBMIT, "fresh attempt"
        )
        return replace(result, actions=(submitted, *result.actions), submitted=True)
