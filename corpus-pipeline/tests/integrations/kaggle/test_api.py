from __future__ import annotations

import os
import sys
import threading
import time

import pytest

from corpus_pipeline.integrations.kaggle.api import KaggleCommandRunner
from corpus_pipeline.integrations.kaggle.errors import KaggleCommandError


def _python_command(source: str) -> list[str]:
    return ["kaggle", "-c", source]


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
