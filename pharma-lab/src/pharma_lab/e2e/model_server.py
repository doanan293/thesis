"""Serve a model on Kaggle GPUs for E2E runs (a laptop CPU is far too slow).

`serve_model` reserves an account, pushes a `rerank-serve` kernel for a reranker or an
`llm-serve` kernel for a chat model, and blocks until the session ends. A watcher reads
the kernel log for the tunnel URL and writes an env file that `pharma-lab e2e run`
sessions source before starting.
"""

from __future__ import annotations

import ast
import json
import re
import secrets
import subprocess
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pharma_agent.domain.llm.models import LlmRole

from pharma_lab.config.paths import GGUF_ROOT, WORK_DIR
from pharma_lab.integrations.kaggle.models import StageName
from pharma_lab.runtime.catalog import ModelKind, require_model
from pharma_lab.runtime.runtime_profiles import RuntimeCandidate

if TYPE_CHECKING:
    from pharma_lab.integrations.kaggle.service import KaggleExecutionContext

SERVE_DIR = WORK_DIR / "serve"
URL_LINE = re.compile(r"\[serve\] url=(https://\S+)")
POLL_SECONDS = 20.0
# Qwen3.5 thinks by default; the pipeline asks for short structured decisions.
CHAT_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}
# A 9B model on T4s generates a long answer in well over the backend's 60 s default.
CHAT_TIMEOUT_SECONDS = 600


def env_path(model: str) -> Path:
    return SERVE_DIR / f"{require_model(model).slug}.env"


def write_request(directory: Path, *, hours: float) -> tuple[Path, str]:
    """A fresh request (new job identity) and a new API key.

    The request is uploaded as a (public) pipeline input dataset, so the key stays in
    a private sibling file that only reaches the private kernel's config.
    """
    if hours <= 0:
        raise ValueError("--hours must be positive")
    directory.mkdir(parents=True, exist_ok=True)
    api_key = secrets.token_urlsafe(32)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"request-{stamp}.json"
    path.write_text(
        json.dumps({"hours": hours, "nonce": secrets.token_hex(8)}), encoding="utf-8"
    )
    key_file = api_key_path(path)
    key_file.write_text(api_key, encoding="utf-8")
    key_file.chmod(0o600)
    return path, api_key


def api_key_path(request_path: Path) -> Path:
    return request_path.with_suffix(".key")


def env_lines(*, model: str, url: str, api_key: str) -> str:
    """Backend settings that point the served model's consumers at the tunnel."""
    if require_model(model).kind is ModelKind.CHAT:
        pairs = [("PHARMA_LLM__TIMEOUT_SECONDS", str(CHAT_TIMEOUT_SECONDS))]
        for role in LlmRole:
            prefix = f"PHARMA_LLM__ROLES__{role.name}__"
            pairs += [
                (prefix + "BASE_URL", f"{url}/v1"),
                (prefix + "API_KEY", api_key),
                (prefix + "MODEL", model),
                (prefix + "EXTRA_BODY", json.dumps(CHAT_EXTRA_BODY)),
            ]
    else:
        pairs = [
            ("PHARMA_RETRIEVAL__RERANK__PROTOCOL", "native_rerank"),
            ("PHARMA_RETRIEVAL__RERANK__BASE_URL", url),
            ("PHARMA_RETRIEVAL__RERANK__API_KEY", api_key),
            ("PHARMA_RETRIEVAL__RERANK__MODEL", model),
            ("PHARMA_RETRIEVAL__RERANK__TIMEOUT_SECONDS", "300"),
        ]
    # JSON values are single-quoted so that `set -a; . file` keeps them intact.
    return "".join(
        f"{name}='{value}'\n" if value.startswith("{") else f"{name}={value}\n"
        for name, value in pairs
    )


def find_url(log_text: str) -> str | None:
    matches = URL_LINE.findall(log_text)
    return matches[-1] if matches else None


@dataclass
class UrlWatcher:
    """Follow a kernel log and write the env file whenever the tunnel URL changes.

    A running kernel's log is only readable in follow mode, so `open_log` starts a
    `kaggle kernels logs -f` process; it is restarted if it ends early.
    """

    open_log: Callable[[], subprocess.Popen[str] | None]
    target: Path
    model: str
    api_key: str
    log: Callable[[str], None]
    interval: float = POLL_SECONDS

    def __post_init__(self) -> None:
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._process: subprocess.Popen[str] | None = None
        self.url: str | None = None

    def feed(self, text: str) -> None:
        url = find_url(text)
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
        while not self._stop.is_set():
            process = self.open_log()
            self._process = process
            if process is not None and process.stdout is not None:
                for line in process.stdout:
                    self.feed(line)
                    if self._stop.is_set():
                        break
                process.terminate()
                process.communicate()
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
        self._thread.join(timeout=10)


def api_key_from_runner(source: str) -> str:
    """The API key embedded in a pushed serve kernel's runner script.

    The runner writes its stage config with `config.write_text("<json>", ...)`.
    """
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "write_text"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "config"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            key = json.loads(node.args[0].value).get("api_key")
            if isinstance(key, str) and key:
                return key
    raise ValueError("the kernel's runner has no serve api_key")


def serve_stage(model: str) -> StageName:
    kind = require_model(model).kind
    if kind is ModelKind.RERANKER:
        return StageName.RERANK_SERVE
    if kind is ModelKind.CHAT:
        return StageName.LLM_SERVE
    raise ValueError(f"--model must select a reranker or a chat model: {model}")


def attach_model(
    *, model: str, reference: str, kaggle_account: str, log: Callable[[str], None]
) -> Path:
    """Serve through a serve kernel that is already running.

    After a local restart the key and env files are gone, but the kernel keeps
    serving: its source holds the key and its log holds the tunnel URL. Blocks until
    the kernel stops.
    """
    from pharma_lab.integrations.kaggle.api import (
        kernel_logs_follow_command,
        kernel_pull_command,
    )
    from pharma_lab.integrations.kaggle.kernel_service import KernelService
    from pharma_lab.integrations.kaggle.models import KernelStatus
    from pharma_lab.integrations.kaggle.service import resolve_execution_context

    context = resolve_execution_context(kaggle_account)
    kernels = KernelService(context.runner, context.owners.execution)
    state = kernels.inspect_state(reference)
    if state.status not in {KernelStatus.QUEUED, KernelStatus.RUNNING}:
        raise ValueError(f"{reference} is not running ({state.status})")
    with tempfile.TemporaryDirectory() as folder:
        context.runner.run(kernel_pull_command(reference, Path(folder)))
        runners = list(Path(folder).glob("*.py"))
        if len(runners) != 1:
            raise ValueError(f"expected one runner script in {reference}")
        api_key = api_key_from_runner(runners[0].read_text(encoding="utf-8"))
    target = env_path(model)
    runner = context.runner
    watcher = UrlWatcher(
        lambda: runner.start(
            kernel_logs_follow_command(reference), capture_output=True
        ),
        target,
        model,
        api_key,
        log,
    )
    log(f"attach kernel={reference} account={kaggle_account}")
    watcher.start()
    try:
        kernels.wait_for_terminal(reference, timeout_seconds=12 * 3600)
    finally:
        watcher.stop()
        target.unlink(missing_ok=True)
    return target


def _serve_profile(
    model: str, budget: int, context: KaggleExecutionContext
) -> RuntimeCandidate:
    """The benchmarked profile of a reranker, or the fixed layout of a chat model."""
    from pharma_lab.integrations.kaggle.auto_profile import ensure_runtime_profile
    from pharma_lab.integrations.kaggle.models import StageName
    from pharma_lab.integrations.kaggle.service import runtime_manifest_sha256

    spec = require_model(model)
    if spec.kind is ModelKind.CHAT:
        if spec.serve_runtime is None:
            raise ValueError(f"{model} has no serve runtime")
        return spec.serve_runtime

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
        kaggle_account=context.profile.name if context.profile else None,
        runtime_sha256=runtime_manifest_sha256(context.runner, context.owners),
        benchmark_runner=no_benchmark,
    )
    if resolution.profile is None:
        raise RuntimeError(f"no Kaggle runtime profile for {model}")
    return resolution.profile.selected


def serve_model(
    *, model: str, hours: float, kaggle_account: str | None, log: Callable[[str], None]
) -> Path:
    """Run one serving session; returns the env file path (removed at the end)."""
    from pharma_lab.integrations.kaggle.api import kernel_logs_follow_command
    from pharma_lab.integrations.kaggle.models import StageRequest
    from pharma_lab.integrations.kaggle.service import (
        resolve_session_contexts,
        run_kaggle_stage,
    )
    from pharma_lab.integrations.kaggle.sessions import reserve_session_account
    from pharma_lab.integrations.kaggle.stages import get_stage_adapter

    stage = serve_stage(model)
    budget = int(hours * 3600) + 1800
    contexts = resolve_session_contexts(kaggle_account)
    profile = _serve_profile(model, budget, contexts[0])
    request_path, api_key = write_request(SERVE_DIR, hours=hours)
    output_dir = SERVE_DIR / "kernels"
    target = env_path(model)
    with reserve_session_account(
        contexts, requested_budget_seconds=budget, log=log
    ) as reserved:
        context = reserved.context
        job = get_stage_adapter(stage).build_job(
            StageRequest(
                stage,
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
        runner = context.runner
        watcher = UrlWatcher(
            lambda: runner.start(
                kernel_logs_follow_command(reference), capture_output=True
            ),
            target,
            model,
            api_key,
            log,
        )
        log(f"serve kernel={reference} account={reserved.profile} hours={hours}")
        watcher.start()
        try:
            run_kaggle_stage(
                stage=stage,
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
            api_key_path(request_path).unlink(missing_ok=True)
    return target
