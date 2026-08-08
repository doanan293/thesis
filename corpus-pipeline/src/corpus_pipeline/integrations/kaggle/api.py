from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from corpus_pipeline.integrations.kaggle.errors import KaggleCommandError


def dataset_metadata(owner: str, slug: str, title: str, public: bool = False) -> dict:
    metadata = {
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
        "enable_internet": str(bool(enable_internet)).lower(),
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
        str(int(timeout_seconds)),
        "-p",
        str(bundle),
    ]


def kernel_status_command(reference: str) -> list[str]:
    return ["kaggle", "kernels", "status", reference]


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
        ".*(jsonl|json|zip)$",
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
        streams = (
            {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "bufsize": 1,
            }
            if capture_output
            else {}
        )
        return subprocess.Popen(command, env=self.environment, text=True, **streams)

    def redact(self, value: str) -> str:
        environment = dict(os.environ)
        environment.update(self.environment or {})
        return _redact(value, environment)

    def run_result(
        self, args: list[str], *, operation: str = "command", target: str | None = None
    ) -> KaggleCommandResult:
        self._validate(args)
        command = [self.executable, *args[1:]]
        if self.dry_run:
            print("DRY RUN: " + " ".join(command), flush=True)
            return KaggleCommandResult(tuple(command), 0, "", "")
        completed = subprocess.run(
            command, env=self.environment, text=True, capture_output=True
        )
        stdout, stderr = completed.stdout or "", completed.stderr or ""
        if completed.returncode or "error:" in stdout.casefold():
            raise KaggleCommandError(
                operation=operation,
                target=target,
                returncode=completed.returncode or 1,
                stdout=self.redact(stdout),
                stderr=self.redact(stderr),
            )
        return KaggleCommandResult(tuple(command), completed.returncode, stdout, stderr)

    def run(self, args: list[str], capture_output: bool = False) -> str:
        del capture_output
        return self.run_result(args).stdout

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


def _redact(value: str, environment: dict[str, str] | None) -> str:
    result = value
    for key in ("KAGGLE_API_TOKEN", "KAGGLE_KEY"):
        secret = (environment or {}).get(key)
        if secret:
            result = result.replace(secret, "<redacted>")
    return result
