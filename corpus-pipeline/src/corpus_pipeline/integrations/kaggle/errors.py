from __future__ import annotations

from enum import StrEnum


class KagglePipelineError(RuntimeError):
    """Base class for failures raised by Kaggle infrastructure."""


class KaggleCommandError(KagglePipelineError):
    def __init__(
        self,
        *,
        operation: str,
        target: str | None,
        returncode: int,
        stdout: str,
        stderr: str,
    ):
        self.operation = operation
        self.target = target
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        detail = (stderr or stdout or "<empty response>").strip()
        subject = f" for {target}" if target else ""
        super().__init__(
            f"Kaggle {operation} failed{subject} "
            f"(exit {self.returncode}): {detail[:2000]}"
        )


class KaggleRemoteStateError(KagglePipelineError):
    pass


class KaggleOutputUnavailable(KagglePipelineError):
    pass


class KaggleDetached(KeyboardInterrupt):
    """Local monitoring stopped while the remote Kaggle kernel keeps running."""

    def __init__(self, reference: str):
        self.reference = reference
        super().__init__(reference)


class ErrorDisposition(StrEnum):
    FATAL = "fatal"
    RETRYABLE = "retryable"


def classify_command_error(error: KaggleCommandError) -> ErrorDisposition:
    detail = f"{error.stdout}\n{error.stderr}".casefold()
    if any(token in detail for token in ("403", "permission", "owner mismatch")):
        return ErrorDisposition.FATAL
    if any(
        token in detail
        for token in ("429", "timeout", "timed out", "500", "502", "503", "504")
    ):
        return ErrorDisposition.RETRYABLE
    return ErrorDisposition.FATAL
