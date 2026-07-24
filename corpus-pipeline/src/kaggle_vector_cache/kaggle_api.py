from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


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
    code_file: str = "kaggle_vector_cache_runner.py",
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


def embeddinggemma_output_command(reference: str, path: Path) -> list[str]:
    return [
        "kaggle",
        "kernels",
        "output",
        reference,
        "-p",
        str(path),
        "-o",
        "--file-pattern",
        ".*embeddinggemma-source/.*",
    ]


def quota_command() -> list[str]:
    return ["kaggle", "quota", "-v"]


@dataclass
class KaggleCommandRunner:
    executable: str = "kaggle"
    environment: dict[str, str] | None = None
    dry_run: bool = False

    def start(self, args: list[str]) -> subprocess.Popen[str] | None:
        if not args or args[0] != "kaggle":
            raise ValueError("Kaggle command must start with 'kaggle'")
        command = [self.executable, *args[1:]]
        if self.dry_run:
            print("DRY RUN: " + " ".join(command), flush=True)
            return None
        return subprocess.Popen(command, env=self.environment, text=True)

    def run(self, args: list[str], capture_output: bool = False) -> str:
        if not args or args[0] != "kaggle":
            raise ValueError("Kaggle command must start with 'kaggle'")
        command = [self.executable, *args[1:]]
        if self.dry_run:
            print("DRY RUN: " + " ".join(command), flush=True)
            return ""
        completed = subprocess.run(
            command,
            env=self.environment,
            check=True,
            text=True,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.STDOUT if capture_output else None,
        )
        return completed.stdout or ""
