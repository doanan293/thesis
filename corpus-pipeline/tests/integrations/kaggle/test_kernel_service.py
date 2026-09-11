import subprocess
from pathlib import Path

import pytest

from corpus_pipeline.integrations.kaggle.errors import (
    KaggleCommandError,
    KaggleDetached,
    KaggleOutputUnavailable,
)
from corpus_pipeline.integrations.kaggle.kernel_service import KernelService
from corpus_pipeline.integrations.kaggle.models import KernelPresence, KernelStatus


class FakeRunner:
    dry_run = False

    def __init__(self, outputs):
        self.outputs = iter(outputs)

    def run(self, args: list[str], capture_output: bool = False) -> str:
        del args, capture_output
        value = next(self.outputs)
        if isinstance(value, BaseException):
            raise value
        return value

    def start(
        self, args: list[str], *, capture_output: bool = False
    ) -> subprocess.Popen[str] | None:
        raise AssertionError("log following is replaced by FakeFollower")

    def redact(self, value: str) -> str:
        return value


class FakeFollower:
    def __init__(self, *_args):
        self.is_running = False
        self.starts = 0
        self.stops = 0

    def start(self):
        self.starts += 1
        self.is_running = True

    def stop(self):
        self.stops += 1
        self.is_running = False

    def poll_failure(self):
        return None


def command_error(detail: str) -> KaggleCommandError:
    return KaggleCommandError(
        operation="status",
        target="owner/missing",
        returncode=1,
        stdout="",
        stderr=detail,
    )


def test_inspect_state_returns_running():
    state = KernelService(
        FakeRunner(['owner/job has status "KernelWorkerStatus.RUNNING"']),
        "owner",
    ).inspect_state("owner/job")

    assert state.presence is KernelPresence.EXISTS
    assert state.status is KernelStatus.RUNNING
    assert state.reference == "owner/job"


@pytest.mark.parametrize(
    "status",
    ["CANCEL_ACKNOWLEDGED", "CANCELED", "CANCELLED"],
)
def test_inspect_state_maps_cancelled_terminal_status_to_error(status):
    state = KernelService(
        FakeRunner([f'owner/job has status "KernelWorkerStatus.{status}"']),
        "owner",
    ).inspect_state("owner/job")

    assert state.presence is KernelPresence.EXISTS
    assert state.status is KernelStatus.ERROR


def test_inspect_state_maps_not_found_to_absent():
    state = KernelService(
        FakeRunner([command_error("404 not found")]), "owner"
    ).inspect_state("owner/missing")

    assert state.presence is KernelPresence.ABSENT
    assert state.status is None


def test_inspect_state_keeps_permission_failure_unknown():
    state = KernelService(
        FakeRunner([command_error("403 permission denied")]), "owner"
    ).inspect_state("owner/private")

    assert state.presence is KernelPresence.UNKNOWN
    assert state.status is None


def test_confirm_missing_kernel_uses_owned_kernel_listing():
    service = KernelService(
        FakeRunner(["ref,title\ndoanvanan0209/other,other\n"]), "owner"
    )

    assert service.confirm_missing("owner/missing") is True


def test_confirm_missing_kernel_returns_false_when_owned():
    service = KernelService(FakeRunner(["ref,title\nowner/missing,missing\n"]), "owner")

    assert service.confirm_missing("owner/missing") is False


def test_wait_for_terminal_returns_complete_and_stops_follower():
    follower = FakeFollower()
    service = KernelService(
        FakeRunner(
            [
                'owner/job has status "KernelWorkerStatus.QUEUED"',
                'owner/job has status "KernelWorkerStatus.RUNNING"',
                'owner/job has status "KernelWorkerStatus.COMPLETE"',
            ]
        ),
        "owner",
        follower_factory=lambda *_args: follower,
        poll_interval_seconds=0.01,
        sleep=lambda _seconds: None,
        monotonic=lambda: 0.0,
    )

    status = service.wait_for_terminal("owner/job", timeout_seconds=60)

    assert status is KernelStatus.COMPLETE
    assert follower.starts == 1
    assert follower.stops == 1


def test_wait_for_terminal_returns_error_without_fetching_log_tail():
    follower = FakeFollower()
    service = KernelService(
        FakeRunner(['owner/job has status "KernelWorkerStatus.ERROR"']),
        "owner",
        follower_factory=lambda *_args: follower,
        sleep=lambda _seconds: None,
        monotonic=lambda: 0.0,
    )

    status = service.wait_for_terminal("owner/job", timeout_seconds=60)

    assert status is KernelStatus.ERROR
    assert follower.starts == 0
    assert follower.stops == 1


def test_wait_for_terminal_translates_local_interrupt_to_detach():
    follower = FakeFollower()
    service = KernelService(
        FakeRunner([KeyboardInterrupt()]),
        "owner",
        follower_factory=lambda *_args: follower,
        monotonic=lambda: 0.0,
    )

    with pytest.raises(KaggleDetached) as caught:
        service.wait_for_terminal("owner/job", timeout_seconds=60)

    assert caught.value.reference == "owner/job"
    assert follower.stops == 1


def test_download_output_retries_transient_command_error():
    service = KernelService(
        FakeRunner(
            [
                command_error("500 temporary failure"),
                command_error("503 service unavailable"),
                "downloaded",
            ]
        ),
        "owner",
        sleep=lambda _seconds: None,
        jitter=lambda: 0.0,
    )

    service.download_output("owner/job", Path("/tmp/output"))


def test_download_output_maps_not_found_to_unavailable():
    service = KernelService(
        FakeRunner([command_error("404 output not found")]), "owner"
    )

    with pytest.raises(KaggleOutputUnavailable):
        service.download_output("owner/job", Path("/tmp/output"))
