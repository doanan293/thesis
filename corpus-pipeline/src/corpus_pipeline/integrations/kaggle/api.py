from __future__ import annotations

import io
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from corpus_pipeline.integrations.kaggle.errors import KaggleCommandError


def dataset_metadata(owner: str, slug: str, title: str, public: bool = False) -> dict:
    metadata: dict[str, object] = {
        "id": f"{owner}/{slug}",
        "title": title,
        "licenses": [{"name": "CC0-1.0"}],
    }
    if public:
        metadata["isPrivate"] = False
    return metadata


def kernel_metadata(
    owner: str,
    kernel_slug: str,
    title: str,
    dataset_sources: list[str],
    *,
    code_file: str = "runner.py",
    enable_internet: bool = False,
) -> dict:
    return {
        "id": f"{owner}/{kernel_slug}",
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": str(enable_internet).lower(),
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": list(dataset_sources),
    }


def dataset_create_command(path: Path, public: bool = False) -> list[str]:
    command = ["kaggle", "datasets", "create", "-p", str(path), "--dir-mode", "skip"]
    if public:
        command.append("--public")
    return command


def dataset_version_command(path: Path, message: str) -> list[str]:
    return [
        "kaggle",
        "datasets",
        "version",
        "-p",
        str(path),
        "-m",
        message,
        "--dir-mode",
        "skip",
    ]


def dataset_status_command(reference: str) -> list[str]:
    return ["kaggle", "datasets", "status", reference, "--format", "json"]


def dataset_list_mine_command(search: str, *, page: int) -> list[str]:
    return [
        "kaggle",
        "datasets",
        "list",
        "--mine",
        "--search",
        search,
        "--csv",
        "--page",
        str(page),
    ]


def config_view_command() -> list[str]:
    return ["kaggle", "config", "view"]


def dataset_metadata_update_command(reference: str, path: Path) -> list[str]:
    return ["kaggle", "datasets", "metadata", reference, "--update", "-p", str(path)]


def dataset_metadata_download_command(reference: str, path: Path) -> list[str]:
    return ["kaggle", "datasets", "metadata", reference, "-p", str(path)]


def dataset_download_command(reference: str, path: Path) -> list[str]:
    return [
        "kaggle",
        "datasets",
        "download",
        reference,
        "-p",
        str(path),
        "--unzip",
        "-o",
    ]


def dataset_files_command(reference: str) -> list[str]:
    return ["kaggle", "datasets", "files", reference, "-v", "--page-size", "100"]


def dataset_file_download_command(
    reference: str, filename: str, path: Path
) -> list[str]:
    return [
        "kaggle",
        "datasets",
        "download",
        reference,
        "-f",
        filename,
        "-p",
        str(path),
        "-o",
        "-q",
    ]


def kernel_push_command(bundle: Path, timeout_seconds: int) -> list[str]:
    return [
        "kaggle",
        "kernels",
        "push",
        "--accelerator",
        "NvidiaTeslaT4",
        "--timeout",
        str(timeout_seconds),
        "-p",
        str(bundle),
    ]


def kernel_status_command(reference: str) -> list[str]:
    return ["kaggle", "kernels", "status", reference]


def kernel_list_mine_command(search: str) -> list[str]:
    return [
        "kaggle",
        "kernels",
        "list",
        "--mine",
        "--search",
        search,
        "--csv",
        "--page-size",
        "100",
    ]


def kernel_logs_command(reference: str) -> list[str]:
    return ["kaggle", "kernels", "logs", reference]


def kernel_logs_follow_command(reference: str) -> list[str]:
    return ["kaggle", "kernels", "logs", "-f", reference]


def kernel_output_command(reference: str, path: Path) -> list[str]:
    return [
        "kaggle",
        "kernels",
        "output",
        reference,
        "-p",
        str(path),
        "-o",
        "--file-pattern",
        ".*(jsonl|json|zip|md|log)$",
    ]


def kernel_manifest_output_command(reference: str, path: Path) -> list[str]:
    return [
        "kaggle",
        "kernels",
        "output",
        reference,
        "-p",
        str(path),
        "-o",
        "--file-pattern",
        r".*checkpoint_manifest\.json$",
    ]


def kernel_output_all_command(reference: str, path: Path) -> list[str]:
    return [
        "kaggle",
        "kernels",
        "output",
        reference,
        "-p",
        str(path),
        "-o",
        "--file-pattern",
        ".*llama-cpp-runtime/.*",
    ]


def quota_command() -> list[str]:
    return ["kaggle", "quota", "-v"]


@dataclass
class KaggleCommandRunner:
    executable: str = "kaggle"
    environment: dict[str, str] | None = None
    dry_run: bool = False

    def start(
        self, args: list[str], *, capture_output: bool = False
    ) -> subprocess.Popen[str] | None:
        self._validate(args)
        command = [self.executable, *args[1:]]
        if self.dry_run:
            print("DRY RUN: " + " ".join(command), flush=True)
            return None
        if capture_output:
            return subprocess.Popen(
                command,
                env=self.environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
            )
        return subprocess.Popen(command, env=self.environment, text=True)

    def redact(self, value: str) -> str:
        environment = dict(os.environ)
        environment.update(self.environment or {})
        return _redact(value, environment)

    def run_result(
        self,
        args: list[str],
        *,
        operation: str = "command",
        target: str | None = None,
        live_output: bool = False,
    ) -> KaggleCommandResult:
        self._validate(args)
        command = [self.executable, *args[1:]]
        if self.dry_run:
            print("DRY RUN: " + " ".join(command), flush=True)
            return KaggleCommandResult(tuple(command), 0, "", "")
        if live_output:
            completed = subprocess.Popen(
                command,
                env=self.environment,
                bufsize=0,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout_parts: list[str] = []
            stderr_parts: list[str] = []
            assert completed.stdout is not None
            assert completed.stderr is not None
            stdout_stream = io.TextIOWrapper(
                completed.stdout, encoding="utf-8", errors="replace", newline=""
            )
            stderr_stream = io.TextIOWrapper(
                completed.stderr, encoding="utf-8", errors="replace", newline=""
            )
            stdout_thread = threading.Thread(
                target=_drain_and_tee,
                args=(stdout_stream, sys.stdout, stdout_parts),
            )
            stderr_thread = threading.Thread(
                target=_drain_and_tee,
                args=(stderr_stream, sys.stderr, stderr_parts),
            )
            stdout_thread.start()
            stderr_thread.start()
            stdout_thread.join()
            stderr_thread.join()
            returncode = completed.wait()
            stdout = "".join(stdout_parts)
            stderr = "".join(stderr_parts)
        else:
            completed = subprocess.run(
                command, env=self.environment, text=True, capture_output=True
            )
            returncode = completed.returncode
            stdout, stderr = completed.stdout or "", completed.stderr or ""
        if returncode or "error:" in stdout.casefold():
            raise KaggleCommandError(
                operation=operation,
                target=target,
                returncode=returncode or 1,
                stdout=self.redact(stdout),
                stderr=self.redact(stderr),
            )
        return KaggleCommandResult(tuple(command), returncode, stdout, stderr)

    def run(
        self,
        args: list[str],
        capture_output: bool = False,
        *,
        live_output: bool = False,
    ) -> str:
        del capture_output
        return self.run_result(args, live_output=live_output).stdout

    @staticmethod
    def _validate(args: list[str]) -> None:
        if not args or args[0] != "kaggle":
            raise ValueError("Kaggle command must start with 'kaggle'")


@dataclass(frozen=True)
class KaggleCommandResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


def _drain_and_tee(stream: TextIO, target: TextIO, parts: list[str]) -> None:
    while chunk := stream.read(1):
        parts.append(chunk)
        target.write(chunk)
        target.flush()


def _redact(value: str, environment: dict[str, str] | None) -> str:
    result = value
    for key in ("KAGGLE_API_TOKEN", "KAGGLE_KEY"):
        secret = (environment or {}).get(key)
        if secret:
            result = result.replace(secret, "<redacted>")
    return result
