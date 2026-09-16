from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import pytest

from seed_pipeline.integrations.kaggle.api import KaggleCommandRunner
from seed_pipeline.integrations.kaggle.errors import KaggleCommandError


def _python_command(source: str) -> list[str]:
    return ["kaggle", "-c", source]


TRANSIENT_ERROR = (
    "http.client.RemoteDisconnected: Remote end closed connection without response"
)


def _flaky_command(marker: Path, failures: int, message: str) -> list[str]:
    """A command that fails `failures` times with `message`, then prints ok."""
    return _python_command(
        "import pathlib, sys\n"
        f"marker = pathlib.Path({str(marker)!r})\n"
        "attempt = int(marker.read_text()) + 1 if marker.exists() else 1\n"
        "marker.write_text(str(attempt))\n"
        f"if attempt <= {failures}:\n"
        f"    sys.stderr.write({message!r})\n"
        "    sys.exit(1)\n"
        "print('ok')\n"
    )


def test_run_result_retries_a_dropped_kaggle_connection(tmp_path):
    """Kaggle drops connections; one blip must not end a scoring run."""
    marker = tmp_path / "attempts"
    sleeps: list[float] = []
    runner = KaggleCommandRunner(executable=sys.executable, sleep=sleeps.append)

    result = runner.run_result(_flaky_command(marker, 1, TRANSIENT_ERROR))

    assert result.stdout == "ok\n"
    assert marker.read_text() == "2"
    assert sleeps == [2.0]


def test_run_result_retries_a_failed_connection(tmp_path):
    """A WSL network blip shows up as NewConnectionError, not as a dropped socket."""
    marker = tmp_path / "attempts"
    sleeps: list[float] = []
    runner = KaggleCommandRunner(executable=sys.executable, sleep=sleeps.append)
    message = (
        "urllib3.exceptions.NewConnectionError: HTTPSConnection(host='api.kaggle.com',"
        " port=443): Failed to establish a new connection: [Errno 22] Invalid argument"
    )

    result = runner.run_result(_flaky_command(marker, 2, message))

    assert result.stdout == "ok\n"
    assert marker.read_text() == "3"
    assert sleeps == [2.0, 4.0]


def test_run_result_does_not_retry_a_real_kaggle_error(tmp_path):
    marker = tmp_path / "attempts"
    sleeps: list[float] = []
    runner = KaggleCommandRunner(executable=sys.executable, sleep=sleeps.append)

    with pytest.raises(KaggleCommandError):
        runner.run_result(_flaky_command(marker, 3, "404 - Not Found"))

    assert marker.read_text() == "1"
    assert sleeps == []


def test_run_result_keeps_default_output_silent(capsys):
    runner = KaggleCommandRunner(executable=sys.executable)

    result = runner.run_result(
        _python_command("import sys; print('out'); print('err', file=sys.stderr)")
    )

    captured = capsys.readouterr()
    assert result.stdout == "out\n"
    assert result.stderr == "err\n"
    assert captured.out == ""
    assert captured.err == ""


def test_run_result_streams_both_outputs_before_process_completion(capsys):
    runner = KaggleCommandRunner(executable=sys.executable)
    finished = threading.Event()

    def invoke() -> None:
        runner.run_result(
            _python_command(
                "import sys,time; print('out', flush=True); "
                "print('err', file=sys.stderr, flush=True); time.sleep(0.5)"
            ),
            live_output=True,
        )
        finished.set()

    thread = threading.Thread(target=invoke)
    thread.start()
    deadline = time.monotonic() + 0.4
    captured = None
    while time.monotonic() < deadline:
        current = capsys.readouterr()
        if "out" in current.out and "err" in current.err:
            captured = current
            break
        time.sleep(0.01)
    assert captured is not None
    assert not finished.is_set()
    thread.join()


def test_live_run_result_preserves_complete_captured_output(capsys):
    runner = KaggleCommandRunner(executable=sys.executable)
    result = runner.run_result(
        _python_command(
            "import sys; sys.stdout.write('a\\rb\\n'); sys.stderr.write('warning\\n')"
        ),
        live_output=True,
    )
    capsys.readouterr()
    assert result.stdout == "a\rb\n"
    assert result.stderr == "warning\n"


def test_live_run_result_keeps_redacted_error_details(capsys):
    runner = KaggleCommandRunner(
        executable=sys.executable,
        environment={**os.environ, "KAGGLE_API_TOKEN": "secret-token"},
    )
    with pytest.raises(KaggleCommandError) as raised:
        runner.run_result(
            _python_command(
                "import sys; print('secret-token'); "
                "print('failed secret-token', file=sys.stderr); sys.exit(2)"
            ),
            live_output=True,
        )
    capsys.readouterr()
    assert "secret-token" not in str(raised.value)
    assert "<redacted>" in str(raised.value)
