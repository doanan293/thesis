"""Serve the reranker on a Kaggle GPU for E2E runs (a laptop CPU is far too slow).

`serve_reranker` reserves an account, pushes a `rerank-serve` kernel and blocks until
the session ends. A watcher reads the kernel log for the tunnel URL and writes an env
file that `pharma-lab e2e run` sessions source before starting.
"""

from __future__ import annotations

import json
import re
import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pharma_lab.config.paths import GGUF_ROOT, WORK_DIR
from pharma_lab.runtime.catalog import ModelKind, require_model

SERVE_DIR = WORK_DIR / "serve"
URL_LINE = re.compile(r"\[serve\] url=(https://\S+)")
POLL_SECONDS = 20.0


def env_path(model: str) -> Path:
    return SERVE_DIR / f"{require_model(model).slug}.env"


def write_request(directory: Path, *, hours: float) -> tuple[Path, str]:
    """A fresh request (new job identity) with a new API key."""
    if hours <= 0:
        raise ValueError("--hours must be positive")
    directory.mkdir(parents=True, exist_ok=True)
    api_key = secrets.token_urlsafe(32)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"request-{stamp}.json"
    path.write_text(
        json.dumps({"hours": hours, "api_key": api_key, "nonce": secrets.token_hex(8)}),
        encoding="utf-8",
    )
    path.chmod(0o600)
    return path, api_key


def env_lines(*, model: str, url: str, api_key: str) -> str:
    return "".join(
        f"{name}={value}\n"
        for name, value in (
            ("PHARMA_RETRIEVAL__RERANK__PROTOCOL", "native_rerank"),
            ("PHARMA_RETRIEVAL__RERANK__BASE_URL", url),
            ("PHARMA_RETRIEVAL__RERANK__API_KEY", api_key),
            ("PHARMA_RETRIEVAL__RERANK__MODEL", model),
            ("PHARMA_RETRIEVAL__RERANK__TIMEOUT_SECONDS", "300"),
        )
    )


def find_url(log_text: str) -> str | None:
    matches = URL_LINE.findall(log_text)
    return matches[-1] if matches else None


@dataclass
class UrlWatcher:
    """Poll a kernel log and write the env file whenever the tunnel URL changes."""

    read_log: Callable[[], str]
    target: Path
    model: str
    api_key: str
    log: Callable[[str], None]
    interval: float = POLL_SECONDS

    def __post_init__(self) -> None:
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.url: str | None = None

    def check(self) -> None:
        url = find_url(self.read_log())
        if url and url != self.url:
            self.url = url
            self.target.parent.mkdir(parents=True, exist_ok=True)
            self.target.write_text(
                env_lines(model=self.model, url=url, api_key=self.api_key),
                encoding="utf-8",
            )
            self.target.chmod(0o600)
            self.log(f"serve url={url} env={self.target}")

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.check()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def serve_reranker(
    *, model: str, hours: float, kaggle_account: str | None, log: Callable[[str], None]
) -> Path:
    """Run one serving session; returns the env file path (removed at the end)."""
    from pharma_lab.integrations.kaggle.auto_profile import ensure_runtime_profile
    from pharma_lab.integrations.kaggle.kernel_service import KernelService
    from pharma_lab.integrations.kaggle.models import StageName, StageRequest
    from pharma_lab.integrations.kaggle.service import (
        resolve_session_contexts,
        run_kaggle_stage,
        runtime_manifest_sha256,
    )
    from pharma_lab.integrations.kaggle.sessions import reserve_session_account
    from pharma_lab.integrations.kaggle.stages import get_stage_adapter

    spec = require_model(model)
    if spec.kind is not ModelKind.RERANKER:
        raise ValueError("--model must select a reranker model")
    budget = int(hours * 3600) + 1800
    contexts = resolve_session_contexts(kaggle_account)
    first = contexts[0]

    def no_benchmark(**_: object) -> object:
        raise RuntimeError(
            f"no Kaggle runtime profile for {model}; run `pharma-lab rerank` first"
        )

    resolution = ensure_runtime_profile(
        workload="rerank",
        benchmark_stage=StageName.RERANK_BENCHMARK.value,
        model=model,
        input_path=Path("unused"),
        gguf_root=GGUF_ROOT,
        budget_seconds=budget,
        dry_run=False,
        force=False,
        kaggle_account=first.profile.name if first.profile else None,
        runtime_sha256=runtime_manifest_sha256(first.runner, first.owners),
        benchmark_runner=no_benchmark,
    )
    if resolution.profile is None:
        raise RuntimeError(f"no Kaggle runtime profile for {model}")
    profile = resolution.profile.selected
    request_path, api_key = write_request(SERVE_DIR, hours=hours)
    output_dir = SERVE_DIR / "kernels"
    target = env_path(model)
    with reserve_session_account(
        contexts, requested_budget_seconds=budget, log=log
    ) as reserved:
        context = reserved.context
        job = get_stage_adapter(StageName.RERANK_SERVE).build_job(
            StageRequest(
                StageName.RERANK_SERVE,
                model,
                request_path,
                output_dir,
                GGUF_ROOT,
                context.owners,
                runtime_profile=profile,
                total_budget_seconds=reserved.budget_seconds,
            )
        )
        reference = (
            f"{context.owners.execution}/{job.stage.value}-{job.identity.sha256[:16]}"
        )
        kernels = KernelService(context.runner, context.owners.execution)
        watcher = UrlWatcher(
            lambda: kernels.log_tail(reference), target, model, api_key, log
        )
        log(f"serve kernel={reference} account={reserved.profile} hours={hours}")
        watcher.start()
        try:
            run_kaggle_stage(
                stage=StageName.RERANK_SERVE,
                model=model,
                input_path=request_path,
                output_dir=output_dir,
                gguf_root=GGUF_ROOT,
                max_runs=1,
                budget_seconds=reserved.budget_seconds,
                runtime_profile=profile,
                kaggle_account=reserved.profile,
            )
        finally:
            watcher.stop()
            target.unlink(missing_ok=True)
            request_path.unlink(missing_ok=True)
    return target
