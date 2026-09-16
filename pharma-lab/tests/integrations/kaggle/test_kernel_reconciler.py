from pathlib import Path

from tests.integrations.kaggle.factories import stage_job

from pharma_lab.integrations.kaggle.errors import KaggleOutputUnavailable
from pharma_lab.integrations.kaggle.kernel_reconciler import KernelReconciler
from pharma_lab.integrations.kaggle.models import (
    ActionVerb,
    KernelPresence,
    KernelRemoteState,
    KernelStatus,
    StageJob,
)


class FakeKernels:
    def __init__(self, remote, *, output_unavailable=False):
        self.remote = remote
        self.output_unavailable = output_unavailable
        self.pushes = []
        self.waits = []
        self.downloads = []

    def reference(self, job: StageJob) -> str:
        return self.remote.reference

    def discover(self, job: StageJob) -> KernelRemoteState:
        return self.remote

    def log_tail(self, reference: str) -> str:
        return f"log tail for {reference}"

    def wait_for_terminal(self, reference, *, timeout_seconds):
        self.waits.append((reference, timeout_seconds))
        self.remote = KernelRemoteState(
            reference, KernelPresence.EXISTS, KernelStatus.COMPLETE
        )
        return KernelStatus.COMPLETE

    def download_output(self, reference, destination):
        self.downloads.append((reference, Path(destination)))
        if self.output_unavailable:
            raise KaggleOutputUnavailable("output unavailable")
        Path(destination).mkdir(parents=True)

    def push(self, bundle, *, timeout_seconds):
        self.pushes.append((Path(bundle), timeout_seconds))


def test_running_kernel_attaches_without_push(tmp_path):
    remote = KernelRemoteState(
        "owner/rerank-legacy", KernelPresence.EXISTS, KernelStatus.RUNNING
    )
    kernels = FakeKernels(remote)
    result = KernelReconciler(kernels).attach_or_recover(
        stage_job(tmp_path), remote, tmp_path, timeout_seconds=60
    )

    assert kernels.pushes == []
    assert result.remote.status is KernelStatus.COMPLETE
    assert result.output_root == tmp_path / "downloaded-output"
    assert [action.verb for action in result.actions] == [
        ActionVerb.ATTACH,
        ActionVerb.DOWNLOAD,
    ]


def test_error_without_output_is_recoverable(tmp_path):
    remote = KernelRemoteState("owner/job", KernelPresence.EXISTS, KernelStatus.ERROR)
    kernels = FakeKernels(remote, output_unavailable=True)

    result = KernelReconciler(kernels).attach_or_recover(
        stage_job(tmp_path), remote, tmp_path, timeout_seconds=60
    )

    assert result.output_root is None
    assert result.remote.status is KernelStatus.ERROR
