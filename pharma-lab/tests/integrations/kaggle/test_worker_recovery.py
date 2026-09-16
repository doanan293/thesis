from pharma_lab.integrations.kaggle.workers.recovery import (
    recoverable_termination,
)
from pharma_lab.integrations.kaggle.workers.runtime import ModelServerExited
from pharma_lab.runtime.client import (
    LlamaCppRequestError,
    LlamaCppResponseError,
)


def test_model_server_exit_is_recoverable_and_bounded():
    payload = recoverable_termination(
        ModelServerExited("x" * 5000, failed_replicas=(1,))
    )

    assert payload == {
        "category": "model_server_unavailable",
        "recoverable": True,
        "error_type": "ModelServerExited",
        "detail": "x" * 4000,
    }


def test_only_retryable_request_errors_are_recoverable():
    assert (
        recoverable_termination(LlamaCppRequestError("HTTP 503", retryable=True))
        is not None
    )
    assert (
        recoverable_termination(LlamaCppRequestError("HTTP 400", retryable=False))
        is None
    )
    assert recoverable_termination(LlamaCppResponseError("bad scores")) is None
