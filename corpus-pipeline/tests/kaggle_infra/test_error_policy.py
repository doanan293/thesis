from corpus_pipeline.integrations.kaggle.errors import (
    ErrorDisposition,
    KaggleCommandError,
    classify_command_error,
)


def command_error(detail: str) -> KaggleCommandError:
    return KaggleCommandError(
        operation="kernel status",
        target="owner/kernel",
        returncode=1,
        stdout="",
        stderr=detail,
    )


def test_permission_error_is_fatal():
    assert (
        classify_command_error(command_error("403 permission 'kernels.get' was denied"))
        is ErrorDisposition.FATAL
    )


def test_rate_limit_error_is_retryable():
    assert (
        classify_command_error(command_error("429 Too Many Requests"))
        is ErrorDisposition.RETRYABLE
    )
