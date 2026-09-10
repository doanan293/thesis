from __future__ import annotations

from corpus_pipeline.integrations.kaggle.workers.runtime import ModelServerExited
from corpus_pipeline.runtime.client import LlamaCppRequestError


def recoverable_termination(error: Exception) -> dict[str, object] | None:
    if not (
        isinstance(error, ModelServerExited)
        or (isinstance(error, LlamaCppRequestError) and error.retryable)
    ):
        return None
    return {
        "category": "model_server_unavailable",
        "recoverable": True,
        "error_type": type(error).__name__,
        "detail": str(error)[:4000],
    }
