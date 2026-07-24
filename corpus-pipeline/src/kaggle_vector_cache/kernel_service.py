from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from kaggle_vector_cache.kaggle_api import (
    KaggleCommandRunner,
    kernel_logs_command,
    kernel_logs_follow_command,
    kernel_manifest_output_command,
    kernel_metadata,
    kernel_output_command,
    kernel_push_command,
    kernel_status_command,
)
from kaggle_vector_cache.models import require_model
from kaggle_vector_cache.parsers import (
    format_compact_elapsed,
    format_elapsed,
    format_timed_log_lines,
    kernel_slug,
    parse_kernel_log_entries,
    parse_kernel_status,
)

CORPUS_DATASET_SLUG = "corpus-pipeline-rag-final"
LLAMA_CPP_DATASET_SLUG = "vector-cache-llama-cpp-cuda-t4"


class KernelService:
    def __init__(
        self,
        runner: KaggleCommandRunner,
        owner: str,
        runtime_owner: str | None = None,
        corpus_owner: str | None = None,
        checkpoint_owner: str | None = None,
    ):
        self.runner = runner
        self.owner = owner
        self.runtime_owner = runtime_owner or owner
        self.corpus_owner = corpus_owner or self.runtime_owner
        self.checkpoint_owner = checkpoint_owner or owner

    def prepare_kernel_bundle(
        self,
        bundle_root: Path,
        model: str,
        checkpoint_dataset: str | None = None,
        autotune_parallel: int = 1,
        max_runs: int = 1,
        total_budget_seconds: int = 21_600,
        export_reserve_seconds: int = 900,
        sample_size: int = 900,
        autotune: bool = True,
    ) -> Path:
        kslug = kernel_slug(model)
        bundle_dir = Path(bundle_root) / kslug
        bundle_dir.mkdir(parents=True, exist_ok=False)

        spec = require_model(model)
        sources = [
            f"{self.runtime_owner}/{LLAMA_CPP_DATASET_SLUG}",
            f"{self.runtime_owner}/{spec.gguf_dataset_slug}",
            f"{self.corpus_owner}/{CORPUS_DATASET_SLUG}",
        ]
        if checkpoint_dataset:
            sources.append(f"{self.checkpoint_owner}/{checkpoint_dataset}")

        meta = kernel_metadata(
            owner=self.owner,
            kernel_slug=kslug,
            title=f"Vector Cache Ingest ({model})",
            dataset_sources=sources,
            code_file="kaggle_vector_cache_runner.py",
            enable_internet=False,
        )
        (bundle_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))

        config_payload = {
            "model": model,
            "max_runs": max_runs,
            "total_budget_seconds": total_budget_seconds,
            "export_reserve_seconds": export_reserve_seconds,
            "sample_size": sample_size,
            "autotune": autotune,
            "autotune_parallel": autotune_parallel,
        }
        config_json_str = json.dumps(config_payload)
        (bundle_dir / "vector_cache_config.json").write_text(
            json.dumps(config_payload, indent=2), encoding="utf-8"
        )

        runner_script = Path("scripts/kaggle_vector_cache_runner.py")
        b64_source = self._build_source_b64()
        if runner_script.exists():
            content = runner_script.read_text(encoding="utf-8")
            if b64_source:
                content = content.replace(
                    'EMBEDDED_SOURCE_B64 = ""', f'EMBEDDED_SOURCE_B64 = "{b64_source}"'
                )
            content = content.replace(
                'EMBEDDED_CONFIG_JSON = ""',
                f"EMBEDDED_CONFIG_JSON = {json.dumps(config_json_str)}",
            )
            (bundle_dir / "kaggle_vector_cache_runner.py").write_text(
                content, encoding="utf-8"
            )
        else:
            (bundle_dir / "kaggle_vector_cache_runner.py").write_text(
                "# Runner script stub\n"
            )

        src_dir = Path("src")
        target_src = bundle_dir / "src"
        if src_dir.exists():
            if target_src.exists():
                shutil.rmtree(target_src)
            shutil.copytree(src_dir, target_src)

        return bundle_dir

    def _build_source_b64(self) -> str:
        import base64
        import io
        import zipfile

        src_dir = Path("src")
        if not src_dir.exists():
            return ""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in src_dir.rglob("*.py"):
                rel = p.relative_to(src_dir)
                zf.write(p, arcname=str(rel))
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def push_kernel(self, bundle_path: Path, timeout_seconds: int = 43200) -> None:
        cmd = kernel_push_command(bundle_path, timeout_seconds=timeout_seconds)
        self.runner.run(cmd)

    def download_manifest(self, reference: str, destination: Path) -> None:
        self.runner.run(kernel_manifest_output_command(reference, destination))

    def download_checkpoint(self, reference: str, destination: Path) -> None:
        self.runner.run(kernel_output_command(reference, destination))

    def forward_kernel_logs(
        self, reference: str, cursor: int = 0, elapsed: float = 0.0
    ) -> tuple[int, bool]:
        output = self.runner.run(
            kernel_logs_command(reference),
            capture_output=True,
        )
        entries = parse_kernel_log_entries(output)
        if cursor > len(entries):
            cursor = 0
        printed = False
        for entry in entries[cursor:]:
            for line in format_timed_log_lines(entry, elapsed):
                print(line, flush=True)
                printed = True
        return len(entries), printed

    def poll_kernel(self, reference: str) -> None:
        if self.runner.dry_run:
            self.runner.run(kernel_status_command(reference), capture_output=True)
            return

        log_process = None
        started_at = time.monotonic()
        last_output_at = started_at
        last_status = None
        log_cursor = 0
        follow_active = False
        log_warning_active = False
        try:
            try:
                log_process = self.runner.start(kernel_logs_follow_command(reference))
                follow_active = log_process is not None
            except OSError as exc:
                print(f"Warning: could not start Kaggle live logs: {exc}", flush=True)

            while True:
                now = time.monotonic()
                elapsed = now - started_at
                output = self.runner.run(
                    kernel_status_command(reference), capture_output=True
                )
                status = parse_kernel_status(output)
                if status != last_status:
                    print(f"[{format_elapsed(elapsed)}] {status}", flush=True)
                    last_status = status
                    last_output_at = now
                elif now - last_output_at >= 60:
                    print(
                        f"[{format_elapsed(elapsed)}] {status} | "
                        f"elapsed={format_compact_elapsed(elapsed)} | "
                        "waiting for new Kaggle worker logs",
                        flush=True,
                    )
                    last_output_at = now
                if follow_active and log_process is not None:
                    return_code = log_process.poll()
                    if return_code is not None:
                        print(
                            f"[{format_elapsed(elapsed)}] Kaggle live logs exited "
                            f"with code {return_code}; switching to snapshot log polling",
                            flush=True,
                        )
                        follow_active = False
                        last_output_at = now
                if status == "COMPLETE":
                    return
                if status == "ERROR":
                    raise RuntimeError(f"Kernel execution failed: {reference}")

                if not follow_active:
                    try:
                        log_cursor, _printed = self.forward_kernel_logs(
                            reference, cursor=log_cursor, elapsed=elapsed
                        )
                        log_warning_active = False
                    except Exception as exc:
                        if not log_warning_active:
                            print(
                                f"Warning: could not poll Kaggle worker logs: {exc}",
                                flush=True,
                            )
                            log_warning_active = True
                time.sleep(15)
        finally:
            if log_process is not None and log_process.poll() is None:
                log_process.terminate()
