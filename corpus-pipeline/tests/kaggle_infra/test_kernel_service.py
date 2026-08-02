from collections import deque

from corpus_pipeline.integrations.kaggle.kernel_service import KernelService


class FakeProcess:
    def __init__(self):
        self.terminated = False

    def terminate(self):
        self.terminated = True


class FakeRunner:
    dry_run = False

    def __init__(self):
        self.statuses = deque(["running", "complete"])
        self.log_process = FakeProcess()

    def start(self, args):
        del args
        return self.log_process

    def run(self, args, **kwargs):
        del kwargs
        if args[1:3] == ["kernels", "status"]:
            return self.statuses.popleft()
        if args[1:3] == ["kernels", "logs"]:
            return "[]"
        return ""


def test_poll_terminates_log_follower_on_completion():
    service = KernelService(FakeRunner(), "owner", poll_interval_seconds=0.01)
    service.poll("owner/kernel", timeout_seconds=1)
