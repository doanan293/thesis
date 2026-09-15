# Kaggle Quota Sessions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `seed rerank --backend kaggle` runs one job as consecutive GPU sessions on the Kaggle account with the most quota, never loses scores when it switches account, runtime profile or job identity, restarts a crashed llama-server inside the kernel, and writes a persistent log per model.

**Architecture:** The local score cache `data/cache/rerank_scores/<model>.jsonl` is the source of truth. A new session runner (`integrations/kaggle/sessions.py`) reads `kaggle quota -v` for every profile before each session, locks the chosen account (`data/work/locks/kaggle-accounts/<accN>.lock`) and calls `run_kaggle_stage(max_runs=1)` for that account. The orchestrator hands every downloaded artifact, complete or partial, to an artifact sink that merges it into the local cache before it publishes a checkpoint, and `CheckpointInheritanceService` offers the local cache as one more checkpoint source before every session. Missing dependency datasets are published by the profile that owns their reference (the `KAGGLE_SHARED_OWNER` profile). The rerank worker wraps each llama-server lifetime in a restart loop (at most 3 restarts). `seed rerank --recover-kernel OWNER/SLUG` merges a finished kernel's scores by record key, independent of job identity.

**Tech Stack:** Python 3.12, uv workspace, Typer, pytest (`--import-mode=importlib`, `filterwarnings = ["error"]`), ruff, pyrefly, Kaggle CLI (`kaggle quota -v`, `kaggle kernels output`), llama.cpp server on Kaggle T4.

**Spec:** seed-pipeline/docs/superpowers/specs/2026-09-15-native-rerank-optimisation-design.md (sections 4.8–4.9)

## Global Constraints

- Execute after Plan A `seed-pipeline/docs/superpowers/plans/2026-09-15-native-rerank-serving.md` (spec 4.1–4.6) is merged. Line numbers below refer to the code before Plan A; locate code by the quoted snippets, not by line number.
- Another session has uncommitted work in `seed-pipeline/src/seed_pipeline/corpus/processing/clean_markdown_corpus.py`, `seed-pipeline/src/seed_pipeline/orchestration/build_corpus.py`, `seed-pipeline/pyproject.toml`, `uv.lock`, `seed-pipeline/data/corpus/*/manifest.json`, `seed-pipeline/data/sources/vietnamese_valid_syllables.json`, `seed-pipeline/tests/corpus/`, `seed-pipeline/tests/orchestration/`, `report/`. Never stage, edit, format or revert these. Stage explicit paths only; never `git add -A`, `git add .` or `git commit -a`.
- Fix lint and type findings in code. No `# noqa`, `# type: ignore`, `# pyrefly: ignore`, ruff ignores or pyrefly error-category changes.
- pytest treats warnings as errors; do not relax `filterwarnings`.
- Tests never call real Kaggle and never read `seed-pipeline/.env`: use fake runners, `KaggleCommandRunner` subclasses, `monkeypatch` on module attributes, and the fakes in `seed-pipeline/tests/integrations/kaggle/factories.py`.
- Tests never write into the real `seed-pipeline/data/`: `tests/conftest.py` redirects `LOCK_DIR` (existing) and `LOGS_DIR` (Task 1).
- Kaggle worker modules (`integrations/kaggle/workers/*`) must not import `seed_pipeline.config.paths` or `pharma_agent` (`tests/integrations/kaggle/test_worker_bundle_imports.py`).
- Quota margin 0.5 h (`1_800` s); minimum session budget 1 h (`3_600` s); at most 3 llama-server restarts per session.
- Account-selection order: most `remaining` GPU hours, ties to the lowest `accN` number; locked accounts are skipped.
- Long-running commands (real Kaggle sessions, benchmarks, `seed data push/pull`) run inside tmux, one window per job. Unit tests and linters run in the foreground.
- **Seed gate** (run in `seed-pipeline/`): `uv run ruff check src tests && uv run ruff format --check src tests && uv run pyrefly check && uv run pytest -q`. Run `uv run ruff format src tests` first; snippets are close to, but not guaranteed to be, formatter output.
- **Commit** from the repository root after each task: `git add` the task's listed paths only, check `git diff --cached --name-status` shows nothing else, then commit with a conventional message (`feat(seed): ...`, `fix(seed): ...`, `docs(seed): ...`) ending with the line `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`. pre-commit runs from the repository root; on failure fix, re-stage the same paths and commit again, never `--no-verify`.

## File Structure

| Area | Files | Responsibility after this plan |
| --- | --- | --- |
| Paths and log | `src/seed_pipeline/config/paths.py`, `src/seed_pipeline/evaluation/rerank_log.py` (new), `tests/conftest.py`, `data/README.md` | `LOGS_DIR`, `rerank_log_path(model)`, timestamped append-only log per model |
| Quota | `src/seed_pipeline/integrations/kaggle/quota.py` (new), `src/seed_pipeline/integrations/kaggle/parsers.py` | Parse `kaggle quota -v`, `AccountQuota`, `select_session_account`, `quota_table` |
| Locks | `src/seed_pipeline/integrations/kaggle/job_lock.py` | `kaggle_account_lock(profile)` besides job and cache locks |
| Worker | `src/seed_pipeline/integrations/kaggle/workers/rerank.py`, `src/seed_pipeline/integrations/kaggle/workers/runtime.py` | Restart llama-server up to 3 times; write a checkpoint artifact from already-scored records |
| Checkpoints | `src/seed_pipeline/integrations/kaggle/checkpoint_inheritance.py`, `src/seed_pipeline/evaluation/rerank_score_cache.py`, `src/seed_pipeline/evaluation/rerank_cache_checkpoint.py` (new) | Best source among local cache and every account checkpoint |
| Dependencies | `src/seed_pipeline/integrations/kaggle/dependencies.py` | Publish a missing dataset with the profile that owns its reference |
| Orchestration | `src/seed_pipeline/integrations/kaggle/models.py`, `src/seed_pipeline/integrations/kaggle/orchestrator.py`, `src/seed_pipeline/integrations/kaggle/service.py` | Artifact sink, `resume_remote`, attached running kernel counts as a run, session contexts, active kernel discovery |
| Sessions | `src/seed_pipeline/integrations/kaggle/sessions.py` (new) | Quota read, account reservation and the session loop |
| Rerank command | `src/seed_pipeline/evaluation/rerank_service.py`, `src/seed_pipeline/cli/commands/rerank.py` | `--kaggle-account auto`, `--max-runs`, `--recover-kernel`, command log lines |
| Docs | `docs/guides/workflow-local-kaggle.md`, `docs/guides/workflow-local-only.md`, `docs/guides/cli-reference.md` | Auto accounts, quota table, tmux, log location, kernel recovery |

All paths in this table are relative to `seed-pipeline/`.

---

### Task 1: Persistent rerank log under `data/work/logs`

**Files:**
- Modify: `seed-pipeline/src/seed_pipeline/config/paths.py`
- Create: `seed-pipeline/src/seed_pipeline/evaluation/rerank_log.py`
- Modify: `seed-pipeline/tests/conftest.py`
- Modify: `seed-pipeline/data/README.md`
- Test: `seed-pipeline/tests/config/test_paths.py`, `seed-pipeline/tests/evaluation/test_rerank_log.py` (new)

**Interfaces:**
- Consumes: `seed_pipeline.runtime.catalog.require_model(model).slug`.
- Produces: `seed_pipeline.config.paths.LOGS_DIR = WORK_DIR / "logs"`; `seed_pipeline.config.paths.rerank_log_path(model: str) -> Path` returning `LOGS_DIR / "rerank" / f"{slug}.log"`; `seed_pipeline.evaluation.rerank_log.RerankLog(path: Path, *, echo: TextIO | None = None, now: Callable[[], datetime] = ...)` whose `__call__(message: str) -> None` appends one `"<iso timestamp> <line>"` per message line and echoes the same lines to `echo`; `open_rerank_log(model: str, *, echo: bool) -> RerankLog` (echo goes to `sys.stderr`).

- [ ] **Step 1: Write the failing tests**

Append to `seed-pipeline/tests/config/test_paths.py`:

```python
def test_rerank_logs_live_under_work(monkeypatch) -> None:
    # tests/conftest.py redirects LOGS_DIR for every test; check the real value.
    monkeypatch.undo()
    assert paths.LOGS_DIR == paths.WORK_DIR / "logs"
    assert paths.rerank_log_path("qwen3-reranker:0.6b-fp16") == (
        paths.WORK_DIR / "logs" / "rerank" / "qwen3_reranker_0_6b_fp16.log"
    )
```

Create `seed-pipeline/tests/evaluation/test_rerank_log.py`:

```python
import io
from datetime import UTC, datetime

from seed_pipeline.config import paths
from seed_pipeline.evaluation.rerank_log import RerankLog, open_rerank_log


def _noon() -> datetime:
    return datetime(2026, 9, 15, 12, 0, tzinfo=UTC)


def test_log_appends_timestamped_lines_and_echoes_them(tmp_path) -> None:
    path = tmp_path / "logs" / "rerank" / "model.log"
    echo = io.StringIO()
    log = RerankLog(path, echo=echo, now=_noon)

    log("account=acc3 budget_seconds=21600")
    log("quota table\nacc1 27.58h")

    expected = (
        "2026-09-15T12:00:00+00:00 account=acc3 budget_seconds=21600\n"
        "2026-09-15T12:00:00+00:00 quota table\n"
        "2026-09-15T12:00:00+00:00 acc1 27.58h\n"
    )
    assert path.read_text(encoding="utf-8") == expected
    assert echo.getvalue() == expected


def test_log_keeps_lines_written_by_earlier_commands(tmp_path) -> None:
    path = tmp_path / "model.log"

    RerankLog(path, now=_noon)("first")
    RerankLog(path, now=_noon)("second")

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [line.split(" ", 1)[1] for line in lines] == ["first", "second"]


def test_open_rerank_log_targets_the_model_log() -> None:
    log = open_rerank_log("qwen3-reranker:0.6b-fp16", echo=False)

    assert log.path == paths.LOGS_DIR / "rerank" / "qwen3_reranker_0_6b_fp16.log"
```

Replace `seed-pipeline/tests/conftest.py` with:

```python
from pathlib import Path

import pytest

from seed_pipeline.config import paths
from seed_pipeline.integrations.kaggle import job_lock


@pytest.fixture(autouse=True)
def isolated_lock_dir(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Locks taken by tests never land in the real data/work/locks."""
    root = tmp_path_factory.mktemp("locks")
    monkeypatch.setattr(job_lock, "LOCK_DIR", root)
    return root


@pytest.fixture(autouse=True)
def isolated_logs_dir(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Command logs written by tests never land in the real data/work/logs."""
    root = tmp_path_factory.mktemp("logs")
    monkeypatch.setattr(paths, "LOGS_DIR", root)
    return root
```

- [ ] **Step 2: Run the tests to verify they fail**

Run (in `seed-pipeline/`): `uv run pytest -q tests/config/test_paths.py tests/evaluation/test_rerank_log.py`
Expected: FAIL — `AttributeError` for `LOGS_DIR` during fixture setup and `ModuleNotFoundError: No module named 'seed_pipeline.evaluation.rerank_log'`.

- [ ] **Step 3: Add the path constant and helper**

In `seed-pipeline/src/seed_pipeline/config/paths.py`, directly below `LOCK_DIR = WORK_DIR / "locks"` add:

```python
LOGS_DIR = WORK_DIR / "logs"
```

and at the end of the file add:

```python
def rerank_log_path(model: str) -> Path:
    return LOGS_DIR / "rerank" / f"{require_model(model).slug}.log"
```

- [ ] **Step 4: Create the log writer**

Create `seed-pipeline/src/seed_pipeline/evaluation/rerank_log.py`:

```python
"""Append-only, timestamped log of every `seed rerank` invocation for one model."""

from __future__ import annotations

import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TextIO

from seed_pipeline.config.paths import rerank_log_path


def _local_now() -> datetime:
    return datetime.now().astimezone()


class RerankLog:
    """One timestamped line per event, kept under data/work so it survives reboots."""

    def __init__(
        self,
        path: Path,
        *,
        echo: TextIO | None = None,
        now: Callable[[], datetime] = _local_now,
    ) -> None:
        self.path = Path(path)
        self._echo = echo
        self._now = now

    def __call__(self, message: str) -> None:
        stamp = self._now().isoformat(timespec="seconds")
        lines = [f"{stamp} {part}" for part in message.splitlines() or [""]]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.writelines(f"{line}\n" for line in lines)
        if self._echo is not None:
            for line in lines:
                print(line, file=self._echo, flush=True)


def open_rerank_log(model: str, *, echo: bool) -> RerankLog:
    return RerankLog(rerank_log_path(model), echo=sys.stderr if echo else None)
```

- [ ] **Step 5: Document the log folder**

In `seed-pipeline/data/README.md` replace the line

```text
└── work/         scratch space: build workspace, locks, bundle-embed input, archive staging
```

with

```text
└── work/         scratch space: build workspace, locks, bundle-embed input, archive staging,
                  logs/rerank/<model>.log (appended by every `seed rerank`)
```

and directly below the line `The first push after a fresh setup needs the owner's confirmation; pull refuses to overwrite files that differ from the archive unless `--force` is given.` add, separated by a blank line, the paragraph:

```markdown
`work/` is never archived. `work/logs/rerank/<model>.log` records, with timestamps, the account and GPU quota of every Kaggle session, the kernel, progress and errors of each `seed rerank` run; it stays on the machine that ran the command and survives restarts.
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest -q tests/config/test_paths.py tests/evaluation/test_rerank_log.py`
Expected: PASS.

- [ ] **Step 7: Run the seed gate**

Run the **Seed gate** from Global Constraints. Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add seed-pipeline/src/seed_pipeline/config/paths.py \
  seed-pipeline/src/seed_pipeline/evaluation/rerank_log.py \
  seed-pipeline/tests/conftest.py seed-pipeline/tests/config/test_paths.py \
  seed-pipeline/tests/evaluation/test_rerank_log.py seed-pipeline/data/README.md
git diff --cached --name-status
git commit -m "feat(seed): keep a persistent rerank log under data/work/logs

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: GPU quota parsing and session account selection

**Files:**
- Create: `seed-pipeline/src/seed_pipeline/integrations/kaggle/quota.py`
- Modify: `seed-pipeline/src/seed_pipeline/integrations/kaggle/parsers.py` (delete the unused `parse_gpu_quota_hours`)
- Test: `seed-pipeline/tests/integrations/kaggle/test_quota.py` (new)

**Interfaces:**
- Consumes: `seed_pipeline.integrations.kaggle.api.quota_command() -> list[str]` (`["kaggle", "quota", "-v"]`); `seed_pipeline.integrations.kaggle.service.KaggleExecutionContext(profile: KaggleAccountProfile | None, owners: OwnerConfiguration, runner: KaggleCommandRunner)`.
- Produces (`seed_pipeline.integrations.kaggle.quota`):
  - `QUOTA_MARGIN_SECONDS = 1_800`, `MINIMUM_SESSION_SECONDS = 3_600`
  - `@dataclass(frozen=True) AccountQuota(profile: str, username: str, remaining_hours: float, total_hours: float, refresh_at: datetime | None)`
  - `@dataclass(frozen=True) SessionAccount(profile: str, budget_seconds: int)`
  - `@dataclass(frozen=True) GpuQuota(remaining_hours: float, total_hours: float, refresh_at: datetime | None)`
  - `parse_gpu_quota(output: str) -> GpuQuota` (raises `RuntimeError` on unusable output)
  - `read_account_quota(context: KaggleExecutionContext) -> AccountQuota`
  - `profile_order(profile: str) -> tuple[int, str]`
  - `select_session_account(quotas: Sequence[AccountQuota], *, locked: frozenset[str], requested_budget_seconds: int, margin_seconds: int = 1800, minimum_seconds: int = 3600) -> SessionAccount | None`
  - `quota_table(quotas: Sequence[AccountQuota], *, locked: frozenset[str] = frozenset()) -> tuple[str, ...]` (header row, then one row per account in `accN` order with remaining, total, `refresh_at` and `locked` yes/no)

The installed Kaggle CLI (`kaggle/api/kaggle_api_extended.py::quota_view_cli`) prints hours as `f"{hours:.2f}h"` and `refreshAt` as `quota_refresh_time.isoformat()` or an empty string; a CLI version warning may precede the CSV header.

- [ ] **Step 1: Write the failing tests**

Create `seed-pipeline/tests/integrations/kaggle/test_quota.py`:

```python
from datetime import datetime

import pytest
from tests.integrations.kaggle.factories import owners

from seed_pipeline.integrations.kaggle.api import KaggleCommandRunner
from seed_pipeline.integrations.kaggle.config import KaggleAccountProfile
from seed_pipeline.integrations.kaggle.quota import (
    AccountQuota,
    SessionAccount,
    parse_gpu_quota,
    quota_table,
    read_account_quota,
    select_session_account,
)
from seed_pipeline.integrations.kaggle.service import KaggleExecutionContext

QUOTA_CSV = (
    "resource,used,remaining,total,refreshAt\n"
    "GPU,2.42h,27.58h,30.00h,2026-09-19T00:00:00\n"
    "TPU,0.00h,20.00h,20.00h,2026-09-19T00:00:00\n"
)
REFRESH = datetime(2026, 9, 19)


class ScriptedRunner(KaggleCommandRunner):
    def __init__(self, output: str):
        super().__init__(environment={})
        self.output = output
        self.commands: list[list[str]] = []

    def run(
        self,
        args: list[str],
        capture_output: bool = False,
        *,
        live_output: bool = False,
    ) -> str:
        del capture_output, live_output
        self.commands.append(list(args))
        return self.output


def _quota(profile: str, remaining: float) -> AccountQuota:
    return AccountQuota(profile, f"user-{profile}", remaining, 30.0, REFRESH)


def test_parse_gpu_quota_reads_hours_and_refresh_time():
    quota = parse_gpu_quota(QUOTA_CSV)

    assert quota.remaining_hours == pytest.approx(27.58)
    assert quota.total_hours == pytest.approx(30.0)
    assert quota.refresh_at == REFRESH


def test_parse_gpu_quota_skips_a_leading_warning_and_a_missing_refresh_time():
    quota = parse_gpu_quota(
        "Warning: a newer version of the kaggle CLI is available\n"
        "resource,used,remaining,total,refreshAt\n"
        "GPU,0.00h,30.00h,30.00h,\n"
    )

    assert quota.remaining_hours == 30.0
    assert quota.refresh_at is None


@pytest.mark.parametrize(
    "output",
    [
        "No quota information available\n",
        "resource,used,remaining,total,refreshAt\nTPU,0.00h,1.00h,1.00h,\n",
        "resource,used,remaining,total,refreshAt\nGPU,1.00h,lots,30.00h,\n",
    ],
)
def test_parse_gpu_quota_rejects_unusable_output(output):
    with pytest.raises(RuntimeError):
        parse_gpu_quota(output)


def test_read_account_quota_uses_the_profile_runner():
    runner = ScriptedRunner(QUOTA_CSV)
    context = KaggleExecutionContext(
        KaggleAccountProfile("acc2", "secondary-user", "token"),
        owners("secondary-user"),
        runner,
    )

    quota = read_account_quota(context)

    assert runner.commands == [["kaggle", "quota", "-v"]]
    assert (quota.profile, quota.username) == ("acc2", "secondary-user")
    assert quota.remaining_hours == pytest.approx(27.58)
    assert quota.refresh_at == REFRESH


def test_select_prefers_the_account_with_the_most_remaining_hours():
    choice = select_session_account(
        [_quota("acc1", 27.58), _quota("acc2", 29.61), _quota("acc3", 30.0)],
        locked=frozenset(),
        requested_budget_seconds=21_600,
    )

    assert choice == SessionAccount("acc3", 21_600)


def test_select_breaks_ties_with_the_lowest_account_number():
    choice = select_session_account(
        [_quota("acc10", 20.0), _quota("acc3", 20.0), _quota("acc2", 20.0)],
        locked=frozenset(),
        requested_budget_seconds=21_600,
    )

    assert choice == SessionAccount("acc2", 21_600)


def test_select_skips_locked_accounts():
    choice = select_session_account(
        [_quota("acc1", 27.58), _quota("acc3", 30.0)],
        locked=frozenset({"acc3"}),
        requested_budget_seconds=21_600,
    )

    assert choice == SessionAccount("acc1", 21_600)


def test_budget_keeps_half_an_hour_of_quota_in_reserve():
    choice = select_session_account(
        [_quota("acc1", 5.0)], locked=frozenset(), requested_budget_seconds=21_600
    )

    assert choice == SessionAccount("acc1", 5 * 3600 - 1800)


def test_no_session_with_less_than_one_hour_of_budget():
    assert select_session_account(
        [_quota("acc1", 1.5)], locked=frozenset(), requested_budget_seconds=21_600
    ) == SessionAccount("acc1", 3600)
    assert (
        select_session_account(
            [_quota("acc1", 1.49)], locked=frozenset(), requested_budget_seconds=21_600
        )
        is None
    )


def test_no_session_when_every_account_is_locked():
    assert (
        select_session_account(
            [_quota("acc1", 30.0)],
            locked=frozenset({"acc1"}),
            requested_budget_seconds=21_600,
        )
        is None
    )


def test_quota_table_lists_refresh_time_and_lock_per_account():
    rows = quota_table(
        [_quota("acc2", 0.2), _quota("acc1", 0.4)], locked=frozenset({"acc2"})
    )

    assert rows[0].split() == [
        "account",
        "username",
        "remaining",
        "total",
        "refresh_at",
        "locked",
    ]
    assert rows[1].split() == [
        "acc1",
        "user-acc1",
        "0.40h",
        "30.00h",
        "2026-09-19T00:00:00",
        "no",
    ]
    assert rows[2].split() == [
        "acc2",
        "user-acc2",
        "0.20h",
        "30.00h",
        "2026-09-19T00:00:00",
        "yes",
    ]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/integrations/kaggle/test_quota.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'seed_pipeline.integrations.kaggle.quota'`.

- [ ] **Step 3: Implement the quota module**

Create `seed-pipeline/src/seed_pipeline/integrations/kaggle/quota.py`:

```python
"""Kaggle GPU quota per account profile and the account choice for one session."""

from __future__ import annotations

import csv
import io
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from seed_pipeline.integrations.kaggle.api import quota_command
from seed_pipeline.integrations.kaggle.service import KaggleExecutionContext

QUOTA_MARGIN_SECONDS = 1_800
MINIMUM_SESSION_SECONDS = 3_600
_PROFILE_NUMBER = re.compile(r"acc([1-9][0-9]*)\Z")


@dataclass(frozen=True)
class AccountQuota:
    profile: str
    username: str
    remaining_hours: float
    total_hours: float
    refresh_at: datetime | None


@dataclass(frozen=True)
class SessionAccount:
    profile: str
    budget_seconds: int


@dataclass(frozen=True)
class GpuQuota:
    remaining_hours: float
    total_hours: float
    refresh_at: datetime | None


def parse_gpu_quota(output: str) -> GpuQuota:
    """Read the GPU row of `kaggle quota -v` (resource,used,remaining,total,refreshAt)."""
    lines = output.splitlines()
    header = next(
        (
            index
            for index, line in enumerate(lines)
            if line.strip().casefold().startswith("resource,")
        ),
        None,
    )
    if header is None:
        raise RuntimeError(
            "Kaggle quota output has no CSV header "
            "'resource,used,remaining,total,refreshAt'"
        )
    for row in csv.DictReader(io.StringIO("\n".join(lines[header:]))):
        if (row.get("resource") or "").strip().upper() != "GPU":
            continue
        refresh = (row.get("refreshAt") or "").strip()
        return GpuQuota(
            remaining_hours=_hours(row.get("remaining")),
            total_hours=_hours(row.get("total")),
            refresh_at=datetime.fromisoformat(refresh) if refresh else None,
        )
    raise RuntimeError("Kaggle quota output did not contain a GPU row")


def _hours(value: str | None) -> float:
    text = (value or "").strip().removesuffix("h")
    try:
        hours = float(text)
    except ValueError as error:
        raise RuntimeError(
            f"Kaggle quota value is not a number of hours: {value!r}"
        ) from error
    if not math.isfinite(hours) or hours < 0:
        raise RuntimeError(f"Kaggle quota value is out of range: {value!r}")
    return hours


def read_account_quota(context: KaggleExecutionContext) -> AccountQuota:
    if context.profile is None:
        raise ValueError(
            "Kaggle quota sessions need an account profile (acc1, acc2, ...)"
        )
    quota = parse_gpu_quota(context.runner.run(quota_command(), capture_output=True))
    return AccountQuota(
        context.profile.name,
        context.profile.username,
        quota.remaining_hours,
        quota.total_hours,
        quota.refresh_at,
    )


def profile_order(profile: str) -> tuple[int, str]:
    match = _PROFILE_NUMBER.fullmatch(profile)
    if match is None:
        raise ValueError(f"invalid Kaggle account profile: {profile!r}")
    return int(match.group(1)), profile


def select_session_account(
    quotas: Sequence[AccountQuota],
    *,
    locked: frozenset[str],
    requested_budget_seconds: int,
    margin_seconds: int = QUOTA_MARGIN_SECONDS,
    minimum_seconds: int = MINIMUM_SESSION_SECONDS,
) -> SessionAccount | None:
    """Pick the unlocked account with the most GPU hours; ties go to the lowest accN.

    The session budget keeps `margin_seconds` of quota in reserve because Kaggle
    reports usage late. No account qualifies when that budget is below
    `minimum_seconds`.
    """
    available = [quota for quota in quotas if quota.profile not in locked]
    if not available:
        return None
    best = min(
        available,
        key=lambda quota: (-quota.remaining_hours, profile_order(quota.profile)),
    )
    usable_seconds = math.floor(best.remaining_hours * 3600) - margin_seconds
    budget = min(requested_budget_seconds, usable_seconds)
    if budget < minimum_seconds:
        return None
    return SessionAccount(best.profile, budget)


def quota_table(
    quotas: Sequence[AccountQuota], *, locked: frozenset[str] = frozenset()
) -> tuple[str, ...]:
    rows = [
        f"{'account':<8} {'username':<24} {'remaining':>10} {'total':>8}  "
        f"{'refresh_at':<25} locked"
    ]
    for quota in sorted(quotas, key=lambda item: profile_order(item.profile)):
        refresh = quota.refresh_at.isoformat() if quota.refresh_at else "unknown"
        rows.append(
            f"{quota.profile:<8} {quota.username:<24} {quota.remaining_hours:>9.2f}h "
            f"{quota.total_hours:>7.2f}h  {refresh:<25} "
            f"{'yes' if quota.profile in locked else 'no'}"
        )
    return tuple(rows)
```

- [ ] **Step 4: Delete the superseded parser**

In `seed-pipeline/src/seed_pipeline/integrations/kaggle/parsers.py` delete the whole function `parse_gpu_quota_hours` (it has no callers: `grep -rn parse_gpu_quota_hours src tests` prints nothing after the deletion). Keep the `csv` and `io` imports; other parsers in the file use them.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -q tests/integrations/kaggle/test_quota.py`
Expected: PASS.

- [ ] **Step 6: Run the seed gate**

Run the **Seed gate**. Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add seed-pipeline/src/seed_pipeline/integrations/kaggle/quota.py \
  seed-pipeline/src/seed_pipeline/integrations/kaggle/parsers.py \
  seed-pipeline/tests/integrations/kaggle/test_quota.py
git diff --cached --name-status
git commit -m "feat(seed): read Kaggle GPU quota and choose the session account

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Kaggle account session lock

**Files:**
- Modify: `seed-pipeline/src/seed_pipeline/integrations/kaggle/job_lock.py`
- Test: `seed-pipeline/tests/integrations/kaggle/test_job_lock.py`

**Interfaces:**
- Consumes: `seed_pipeline.integrations.kaggle.config.ACCOUNT_NAME` (regex `acc([1-9][0-9]*)\Z`); `LOCK_DIR` (redirected by `tests/conftest.py`).
- Produces (`seed_pipeline.integrations.kaggle.job_lock`):
  - `class KaggleAccountBusy(RuntimeError)` with attribute `profile: str`.
  - `kaggle_account_lock(profile: str, *, lock_root: Path | None = None) -> AbstractContextManager[str]` — non-blocking flock on `<lock_root>/kaggle-accounts/<profile>.lock`; entering yields the profile; entering while another handle holds it raises `KaggleAccountBusy` immediately; calling it with a name that is not `accN` raises `ValueError`.
  - `kaggle_job_lock` and `kaggle_cache_lock` keep their signatures and messages.

- [ ] **Step 1: Write the failing tests**

In `seed-pipeline/tests/integrations/kaggle/test_job_lock.py` change the import block to:

```python
from seed_pipeline.integrations.kaggle.job_lock import (
    KaggleAccountBusy,
    kaggle_account_lock,
    kaggle_cache_lock,
    kaggle_job_lock,
    lock_file_name,
)
```

and append:

```python
def test_account_lock_uses_one_file_per_account(tmp_path: Path) -> None:
    lock_root = tmp_path / "locks"

    with kaggle_account_lock("acc2", lock_root=lock_root) as profile:
        assert profile == "acc2"
        assert (lock_root / "kaggle-accounts" / "acc2.lock").is_file()


def test_two_jobs_cannot_hold_the_same_account(tmp_path: Path) -> None:
    lock_root = tmp_path / "locks"
    entered = threading.Event()
    release = threading.Event()

    def first_job() -> None:
        with kaggle_account_lock("acc3", lock_root=lock_root):
            entered.set()
            release.wait(2)

    thread = threading.Thread(target=first_job)
    thread.start()
    assert entered.wait(2)
    try:
        with (
            pytest.raises(KaggleAccountBusy, match="acc3"),
            kaggle_account_lock("acc3", lock_root=lock_root),
        ):
            raise AssertionError("a held account must not be entered")
        with kaggle_account_lock("acc1", lock_root=lock_root):
            pass
    finally:
        release.set()
        thread.join()

    with kaggle_account_lock("acc3", lock_root=lock_root):
        pass


@pytest.mark.parametrize("profile", ["auto", "../acc1", "acc0", ""])
def test_account_lock_rejects_names_that_are_not_profiles(
    profile: str, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="invalid Kaggle account profile"):
        kaggle_account_lock(profile, lock_root=tmp_path)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/integrations/kaggle/test_job_lock.py`
Expected: FAIL with `ImportError: cannot import name 'KaggleAccountBusy'`.

- [ ] **Step 3: Implement the account lock**

Replace everything in `seed-pipeline/src/seed_pipeline/integrations/kaggle/job_lock.py` from the first import to the end of `_acquire` with the code below, keep `lock_file_name`, `kaggle_job_lock` and `kaggle_cache_lock` unchanged, and append `kaggle_account_lock` and `_account_lock` at the end of the file:

```python
from __future__ import annotations

import fcntl
import os
from collections.abc import Generator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from pathlib import Path
from typing import Literal

from seed_pipeline.config.paths import DATA_DIR, LOCK_DIR
from seed_pipeline.integrations.kaggle.config import ACCOUNT_NAME

LockKind = Literal["job", "cache", "publish"]
MAX_LOCK_FILE_NAME_BYTES = 255
ACCOUNT_LOCK_DIRNAME = "kaggle-accounts"


class KaggleAccountBusy(RuntimeError):
    """Another local command holds the session lock of this Kaggle account."""

    def __init__(self, profile: str):
        super().__init__(
            f"Kaggle account {profile} already runs a session for another local job"
        )
        self.profile = profile


class _LockHeldError(Exception):
    """A non-blocking flock found the lock taken."""


# (lock_file_name stays here unchanged)


@contextmanager
def _flock(
    lock_path: Path, content: str, *, non_blocking: bool
) -> Generator[None, None, None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        flags = fcntl.LOCK_EX | (fcntl.LOCK_NB if non_blocking else 0)
        try:
            fcntl.flock(handle.fileno(), flags)
        except BlockingIOError as exc:
            raise _LockHeldError(str(lock_path)) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(content)
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _acquire(
    target: Path,
    *,
    kind: LockKind,
    lock_root: Path,
    non_blocking: bool,
) -> Generator[Path, None, None]:
    canonical = str(Path(target).resolve(strict=False))
    lock_path = Path(lock_root) / lock_file_name(target, kind)
    with ExitStack() as stack:
        # Only the acquisition is translated; errors from the body pass through.
        try:
            stack.enter_context(_flock(lock_path, canonical, non_blocking=non_blocking))
        except _LockHeldError as exc:
            raise RuntimeError(
                f"Local target {canonical} already has an active Kaggle job; "
                "wait for it to finish or choose a different run/job"
            ) from exc
        yield Path(target)
```

Remove the `# (lock_file_name stays here unchanged)` comment line; it only marks where the existing function stays. Append:

```python
def kaggle_account_lock(
    profile: str,
    *,
    lock_root: Path | None = None,
) -> AbstractContextManager[str]:
    """Hold the session lock of one Kaggle account, failing at once if it is taken.

    Parallel `seed rerank` commands for different models share the account pool;
    this lock keeps each account to one GPU session started from this machine.
    """
    if ACCOUNT_NAME.fullmatch(profile) is None:
        raise ValueError(f"invalid Kaggle account profile: {profile!r}")
    root = LOCK_DIR if lock_root is None else lock_root
    return _account_lock(profile, Path(root) / ACCOUNT_LOCK_DIRNAME / f"{profile}.lock")


@contextmanager
def _account_lock(profile: str, lock_path: Path) -> Generator[str, None, None]:
    with ExitStack() as stack:
        try:
            stack.enter_context(
                _flock(lock_path, f"{profile} pid={os.getpid()}\n", non_blocking=True)
            )
        except _LockHeldError as exc:
            raise KaggleAccountBusy(profile) from exc
        yield profile
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q tests/integrations/kaggle/test_job_lock.py`
Expected: PASS (the existing job and cache lock tests pass unchanged).

- [ ] **Step 5: Run the seed gate**

Run the **Seed gate**. Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add seed-pipeline/src/seed_pipeline/integrations/kaggle/job_lock.py \
  seed-pipeline/tests/integrations/kaggle/test_job_lock.py
git diff --cached --name-status
git commit -m "feat(seed): lock a Kaggle account for the length of one session

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

### Task 4: Restart llama-server inside the rerank worker

**Files:**
- Modify: `seed-pipeline/src/seed_pipeline/integrations/kaggle/workers/runtime.py` (`managed_model_servers`)
- Modify: `seed-pipeline/src/seed_pipeline/integrations/kaggle/workers/rerank.py`
- Test: `seed-pipeline/tests/integrations/kaggle/test_worker_runtime.py`, `seed-pipeline/tests/integrations/kaggle/test_worker_rerank.py`

**Interfaces:**
- Consumes: `workers.recovery.recoverable_termination(error: Exception) -> dict[str, object] | None`; the native scoring block Plan A leaves in `run_rerank_worker` (one `/v1/rerank` request per query group, scheduled by `stream_map_ordered`, committed by `commit(records)`).
- Produces:
  - `managed_model_servers(config: dict, *, telemetry: object | None = None, restart_index: int = 0, close_telemetry: bool = True)` — server logs are written as `server-<i>.log` for `restart_index == 0` and `server-<i>.restart-<n>.log` otherwise; telemetry is closed and written only when `close_telemetry` is true.
  - `workers.rerank.MAX_SERVER_RESTARTS = 3`
  - `@dataclass(frozen=True) workers.rerank.ServerLifetimes(restarts: int, termination: dict[str, object] | None)`
  - `workers.rerank.score_with_server_restarts(score_once: Callable[[int], None], *, deadline: float, emit: Callable[[str], None], max_restarts: int = MAX_SERVER_RESTARTS, clock: Callable[[], float] = time.monotonic) -> ServerLifetimes`
  - The rerank artifact manifest gains `runtime.server_restarts: int`; a sealed partial artifact's `runtime.termination` gains `server_restarts`.

Why: kernel `doanvanan0209/rerank-5f22fcadeede1072` sealed 94,950/300,000 pairs at 7,659 s after one retryable llama-server failure while almost 4 h of budget remained.

- [ ] **Step 1: Write the failing runtime test**

Append to `seed-pipeline/tests/integrations/kaggle/test_worker_runtime.py` (the file already imports `io` and `managed_model_servers`):

```python
def test_managed_model_servers_can_leave_telemetry_open_and_name_restart_logs(
    monkeypatch, tmp_path
):
    from seed_pipeline.integrations.kaggle.workers import runtime
    from seed_pipeline.runtime import client

    class QuietProcess:
        def __init__(self, output: bytes):
            self.stdout = io.BytesIO(output)
            self.returncode = None
            self.pid = 2000

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0

        def wait(self, timeout=None):
            return self.returncode

    class RecordingTelemetry:
        def __init__(self):
            self.calls: list[str] = []

        def register_server_processes(self, processes):
            self.calls.append("register")

        def close(self):
            self.calls.append("close")

        def write_report(self):
            self.calls.append("write_report")

    model_path = tmp_path / "model.gguf"
    model_path.touch()
    manifest_path = tmp_path / "runtime_manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        runtime,
        "find_unique",
        lambda _root, pattern: model_path if pattern == "*.gguf" else manifest_path,
    )
    monkeypatch.setattr(runtime, "materialize_runtime", lambda *_args: tmp_path)
    monkeypatch.setattr(
        runtime.subprocess,
        "Popen",
        lambda *_args, **_kwargs: QuietProcess(b"restarted server\n"),
    )
    monkeypatch.setattr(client.LlamaCppClient, "health", lambda _self: None)
    telemetry = RecordingTelemetry()
    config = {
        "model": "qwen3-reranker:0.6b-fp16",
        "runtime_overrides": {
            "server_slots": 2,
            "context_per_slot": 4096,
            "logical_batch_size": 4096,
            "physical_batch_size": 2048,
        },
        "output_dir": str(tmp_path / "output"),
    }

    with managed_model_servers(
        config, telemetry=telemetry, restart_index=2, close_telemetry=False
    ) as servers:
        assert servers

    assert telemetry.calls == ["register"]
    output = tmp_path / "output"
    assert (output / "server-0.restart-2.log").read_text(encoding="utf-8") == (
        "restarted server\n"
    )
    assert not (output / "server-0.log").exists()
```

If Plan A renamed keys of `runtime_overrides` in `test_managed_model_servers_wraps_mid_run_exit_with_diagnostics`, use the same keys here.

- [ ] **Step 2: Write the failing worker tests**

In `seed-pipeline/tests/integrations/kaggle/test_worker_rerank.py` add `import threading` to the imports and append:

```python
_NATIVE_OVERRIDES = {
    "server_slots": 1,
    "concurrency": 1,
    "request_batch_size": 1,
    "context_per_slot": 4096,
    "logical_batch_size": 4096,
    "physical_batch_size": 2048,
}


def _native_config(tmp_path: Path, query_count: int) -> dict:
    config = _config(tmp_path, query_count=query_count, candidates_per_query=1)
    config["model"] = "qwen3-reranker:0.6b-fp16"
    config["protocol"] = "native_rerank"
    config["runtime_overrides"] = dict(_NATIVE_OVERRIDES)
    return config


class _CountingServerManager:
    def __init__(self):
        self.restart_indexes: list[int] = []

    @contextmanager
    def __call__(
        self, _config, *, telemetry=None, restart_index=0, close_telemetry=True
    ):
        del telemetry, close_telemetry
        self.restart_indexes.append(restart_index)
        yield [SimpleNamespace(base_url="http://127.0.0.1:11434")]


class _FlakyQuery:
    """Drops the connection for the first `failures` requests of one query."""

    def __init__(self, query: str, failures: int):
        self.query = query
        self.failures = failures
        self._lock = threading.Lock()

    def rerank_native(self, query, documents, _model):
        with self._lock:
            if query == self.query and self.failures > 0:
                self.failures -= 1
                raise LlamaCppRequestError("connection refused", retryable=True)
        return [0.5 for _document in documents]


def test_rerank_worker_restarts_the_server_and_scores_the_missing_pairs(tmp_path):
    config = _native_config(tmp_path, query_count=3)
    manager = _CountingServerManager()
    client = _FlakyQuery("query 1", failures=2)
    messages: list[str] = []

    artifact = run_rerank_worker(
        config,
        server_manager=manager,
        client_factory=lambda _url: client,
        emit=messages.append,
        clock=lambda: 0.0,
    )

    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    assert artifact.completion.complete == 3
    assert manager.restart_indexes == [0, 1, 2]
    assert manifest["runtime"]["server_restarts"] == 2
    assert "termination" not in manifest["runtime"]
    assert any("restart 2/3" in message for message in messages)


def test_rerank_worker_seals_a_partial_artifact_after_three_restarts(tmp_path):
    config = _native_config(tmp_path, query_count=1)
    manager = _CountingServerManager()
    client = _FlakyQuery("query 0", failures=10)

    artifact = run_rerank_worker(
        config,
        server_manager=manager,
        client_factory=lambda _url: client,
        emit=lambda _message: None,
        clock=lambda: 0.0,
    )

    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    assert manager.restart_indexes == [0, 1, 2, 3]
    assert artifact.completion.complete == 0
    termination = manifest["runtime"]["termination"]
    assert termination["category"] == "model_server_unavailable"
    assert termination["server_restarts"] == 3
    assert manifest["runtime"]["server_restarts"] == 3
```

The first test is deterministic under any client concurrency: only `query 1` fails, each server lifetime ends at its first failure, and every pending pair is retried in the next lifetime. If Plan A changed the `rerank_native` signature, give `_FlakyQuery.rerank_native` the same signature as `_SucceedOnceThenDisconnect.rerank_native` in this file, and if Plan A's `_config` no longer takes `candidates_per_query`, call it the way the other native tests in this file do.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest -q tests/integrations/kaggle/test_worker_runtime.py tests/integrations/kaggle/test_worker_rerank.py`
Expected: FAIL — `TypeError: managed_model_servers() got an unexpected keyword argument 'restart_index'`, and in the worker tests `restart_indexes == [0]` with a partial artifact.

- [ ] **Step 4: Make server lifetimes restartable in `managed_model_servers`**

In `seed-pipeline/src/seed_pipeline/integrations/kaggle/workers/runtime.py` change the signature to:

```python
@contextmanager
def managed_model_servers(
    config: dict,
    *,
    telemetry: object | None = None,
    restart_index: int = 0,
    close_telemetry: bool = True,
) -> Generator[list[WorkerServer], None, None]:
    """Start one local llama.cpp server per configured replica on Kaggle.

    A caller that restarts the servers passes `restart_index` so each lifetime keeps
    its own server logs, and `close_telemetry=False` so sampling continues across
    lifetimes; that caller then closes the telemetry itself.
    """
```

and replace the tail of the `finally:` block, starting at `output_dir = Path(config.get("output_dir", "/kaggle/working/artifact"))`, with:

```python
        output_dir = Path(config.get("output_dir", "/kaggle/working/artifact"))
        suffix = "" if restart_index == 0 else f".restart-{restart_index}"
        for index, collector in enumerate(collectors):
            collector.write(output_dir / f"server-{index}{suffix}.log")
        if telemetry is not None and close_telemetry:
            close = getattr(telemetry, "close", None)
            write_report = getattr(telemetry, "write_report", None)
            if callable(close):
                close()
            if callable(write_report):
                write_report()
```

`load_cloud_artifact` already lists `server-*.log` as diagnostics, so restart logs travel with the artifact.

- [ ] **Step 5: Add the restart loop to the rerank worker**

In `seed-pipeline/src/seed_pipeline/integrations/kaggle/workers/rerank.py` add `from dataclasses import dataclass` to the imports and add below `_emit_progress`:

```python
MAX_SERVER_RESTARTS = 3


@dataclass(frozen=True)
class ServerLifetimes:
    restarts: int
    termination: dict[str, object] | None


def score_with_server_restarts(
    score_once: Callable[[int], None],
    *,
    deadline: float,
    emit: Callable[[str], None],
    max_restarts: int = MAX_SERVER_RESTARTS,
    clock: Callable[[], float] = time.monotonic,
) -> ServerLifetimes:
    """Run scoring passes, one per llama-server lifetime.

    `score_once(restart_index)` starts the servers (waiting for /health), scores the
    pairs that are still missing and returns. After a recoverable model-server failure
    a new lifetime starts, at most `max_restarts` times and never after the deadline.
    `termination` describes the last failure when no restart is left.
    """
    restarts = 0
    while True:
        try:
            score_once(restarts)
        except Exception as error:
            termination = recoverable_termination(error)
            if termination is None:
                raise
            if restarts >= max_restarts or clock() >= deadline:
                return ServerLifetimes(
                    restarts, termination | {"server_restarts": restarts}
                )
            restarts += 1
            emit(
                f"model server unavailable ({type(error).__name__}); "
                f"restart {restarts}/{max_restarts}"
            )
            continue
        return ServerLifetimes(restarts, None)
```

Then restructure the server branch of `run_rerank_worker`. Replace the block that starts with `termination = None` and ends just before `journal_path.parent.mkdir(parents=True, exist_ok=True)` with the code below. Inside `score_once`, keep the body of `native_operation`, `run_native` and the `per_client` concurrency expression exactly as Plan A left them; the only changes are that pending pairs are recomputed from `existing` at the start of every lifetime, the server manager receives `restart_index` and `close_telemetry=False`, and the whole lifetime runs through `score_with_server_restarts`. The snippet shows the pre-Plan-A native body for reference:

```python
lifetimes = ServerLifetimes(0, None)
if score_pair is None and missing:
    server_config = dict(config)
    runtime_overrides = server_config.get("runtime_overrides")
    if not isinstance(runtime_overrides, dict):
        raise ValueError("runtime_overrides is required")
    server_config["runtime_overrides"] = runtime_overrides
    if client_factory is None:
        from seed_pipeline.runtime.client import LlamaCppClient

        client_factory = LlamaCppClient
    make_client = client_factory

    def score_once(restart_index: int) -> None:
        pending = [pair for pair in pairs if _pair_key(pair) not in existing]
        if not pending:
            return
        with server_manager(
            server_config,
            telemetry=telemetry,
            restart_index=restart_index,
            close_telemetry=False,
        ) as servers:
            clients = [make_client(server.base_url) for server in servers]
            resources = list(enumerate(clients))
            per_client = max(1, int(runtime_overrides["concurrency"]))
            grouped: dict[tuple[str, str], list[Pair]] = {}
            for pair in pending:
                query, _text = details[_pair_key(pair)]
                grouped.setdefault((pair["query_id"], query), []).append(pair)
            groups = [
                (query_id, query, group) for (query_id, query), group in grouped.items()
            ]

            async def native_operation(resource, _index, item):
                server_index, client = resource
                query_id, query, group = item
                started_operation = clock()
                scores = await asyncio.to_thread(
                    client.rerank_native,
                    query,
                    [details[_pair_key(pair)][1] for pair in group],
                    model,
                )
                if len(scores) != len(group):
                    raise ValueError("reranker returned an unexpected score count")
                telemetry.record_operation(
                    server_index,
                    len(group),
                    sum(
                        len(query) + len(details[_pair_key(pair)][1]) for pair in group
                    ),
                    max(0.0, clock() - started_operation),
                    "ok",
                    0,
                )
                return (query_id, group, scores)

            async def run_native():
                return await stream_map_ordered(
                    groups,
                    resources,
                    per_client,
                    native_operation,
                    deadline,
                    on_completed=lambda batch: commit(
                        [
                            _score_record(pair, score)
                            for _index, value in batch
                            for pair, score in zip(value[1], value[2], strict=True)
                        ]
                    ),
                )

            scheduled = asyncio.run(run_native())
            if scheduled.stopped_early:
                emit(f"budget exhausted pairs={len(existing)}/{len(pairs)}")

    try:
        lifetimes = score_with_server_restarts(score_once, deadline=deadline, emit=emit)
    except BaseException:
        telemetry.close()
        raise
termination = lifetimes.termination
```

Directly below `runtime_summary = telemetry.summary()` add:

```python
    runtime_summary["server_restarts"] = lifetimes.restarts
```

The existing `if termination is not None:` block that stores `runtime_summary["termination"]` and emits `recoverable model-server termination; checkpointed pairs=...` stays unchanged. If the pre-Plan-A `else:` completion branch still exists (Plan A deletes it), delete it together with this change.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest -q tests/integrations/kaggle/test_worker_runtime.py tests/integrations/kaggle/test_worker_rerank.py tests/integrations/kaggle/test_worker_bundle_imports.py`
Expected: PASS, including the existing `test_rerank_worker_seals_partial_artifact_after_server_failure` (it now restarts three times before sealing the same partial artifact).

- [ ] **Step 7: Run the seed gate**

Run the **Seed gate**. Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add seed-pipeline/src/seed_pipeline/integrations/kaggle/workers/runtime.py \
  seed-pipeline/src/seed_pipeline/integrations/kaggle/workers/rerank.py \
  seed-pipeline/tests/integrations/kaggle/test_worker_runtime.py \
  seed-pipeline/tests/integrations/kaggle/test_worker_rerank.py
git diff --cached --name-status
git commit -m "fix(seed): restart llama-server up to three times before sealing a partial rerank artifact

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Local score cache as a checkpoint source

**Files:**
- Modify: `seed-pipeline/src/seed_pipeline/integrations/kaggle/workers/rerank.py` (add `write_rerank_checkpoint`)
- Modify: `seed-pipeline/src/seed_pipeline/evaluation/rerank_score_cache.py` (add `available_records`)
- Create: `seed-pipeline/src/seed_pipeline/evaluation/rerank_cache_checkpoint.py`
- Modify: `seed-pipeline/src/seed_pipeline/integrations/kaggle/checkpoint_inheritance.py`
- Test: `seed-pipeline/tests/integrations/kaggle/test_checkpoint_inheritance.py`, `seed-pipeline/tests/evaluation/test_rerank_cache_checkpoint.py` (new)

**Interfaces:**
- Consumes: `AppendOnlyJournal.open(path, identity, key_fn, fingerprint_fn, seed_path=None)`, `artifact_from_output(...)`, `JobIdentity.reuse_sha256` (Plan A: excludes `runtime_parameters.runtime_profile`), `CheckpointService.publish_if_better(job, artifact, current) -> CheckpointState`, `kaggle_cache_lock(path)`.
- Produces:
  - `workers.rerank.write_rerank_checkpoint(records: Sequence[dict], *, identity: JobIdentity, total: int, output_dir: Path) -> CloudArtifact` — writes `rerank_scores.jsonl`, `rerank_scores.journal.jsonl` (journal identity `{"reuse_sha256": identity.reuse_sha256}`, the same as the worker) and `manifest.json`.
  - `RerankScoreCache.available_records(candidate_data_path: Path, reranker: str) -> list[dict[str, Any]]` — stored records of the candidate pairs that already have a score, in key order.
  - `seed_pipeline.evaluation.rerank_cache_checkpoint.RerankCacheCheckpoint(cache_path: Path)` with `inspect(job: StageJob, *, download_root: Path) -> CheckpointState` (`reference=None`; `artifact=None` when no pair is scored).
  - `checkpoint_inheritance.LOCAL_SOURCE = "local"`; `class LocalCheckpointSource(Protocol)` with `inspect(self, job: StageJob, /, *, download_root: Path) -> CheckpointState`.
  - `CheckpointInheritanceService(*, target_profile, target, candidates, temp_root, local: LocalCheckpointSource | None = None)`; `resolve(job, target_state, *, check_only: bool, include_profiles: bool = True) -> CheckpointInheritanceResult`. The local source is read first, so it wins ties against account checkpoints; the target is published only when a source has strictly more pairs. Action reasons read `checkpoint inherited local -> acc3: N/T pairs`.

- [ ] **Step 1: Write the failing inheritance tests**

In `seed-pipeline/tests/integrations/kaggle/test_checkpoint_inheritance.py` replace `_service` with:

```python
def _service(target, sources, tmp_path, local=None):
    return CheckpointInheritanceService(
        target_profile="acc3",
        target=target,
        candidates=tuple(
            ProfileCheckpointService(name, service) for name, service in sources
        ),
        temp_root=tmp_path,
        local=local,
    )
```

and append:

```python
class FakeLocalSource:
    def __init__(self, state):
        self.state = state
        self.roots = []

    def inspect(self, job, *, download_root):
        self.roots.append(download_root)
        return self.state


def test_resolve_publishes_the_local_cache_when_it_has_the_most_pairs(tmp_path):
    target = FakeCheckpointService(
        _state(10, 3, "acc3/checkpoint"),
        reference="acc3/checkpoint",
        revalidated=_state(10, 7, "acc3/checkpoint"),
    )
    acc1 = FakeCheckpointService(_state(10, 5, "acc1/checkpoint"))
    local = FakeLocalSource(_state(10, 7))

    result = _service(
        target, (("acc1", acc1), ("acc3", target)), tmp_path, local=local
    ).resolve(stage_job(tmp_path), target.state, check_only=False)

    assert target.published[0][0] is local.state.artifact
    assert "local -> acc3" in result.actions[0].reason
    assert "7/10" in result.actions[0].reason
    assert result.state.completion.complete == 7


def test_resolve_prefers_an_account_checkpoint_with_more_pairs_than_local(tmp_path):
    target = FakeCheckpointService(
        _state(10, 0), revalidated=_state(10, 6, "acc3/checkpoint")
    )
    acc1 = FakeCheckpointService(_state(10, 6, "acc1/checkpoint"))
    local = FakeLocalSource(_state(10, 4))

    result = _service(
        target, (("acc1", acc1), ("acc3", target)), tmp_path, local=local
    ).resolve(stage_job(tmp_path), target.state, check_only=False)

    assert target.published[0][0] is acc1.state.artifact
    assert "acc1 -> acc3" in result.actions[0].reason


def test_resolve_never_replaces_the_target_with_fewer_pairs(tmp_path):
    target = FakeCheckpointService(_state(10, 8, "acc3/checkpoint"))
    acc1 = FakeCheckpointService(_state(10, 6, "acc1/checkpoint"))
    local = FakeLocalSource(_state(10, 5))

    result = _service(
        target, (("acc1", acc1), ("acc3", target)), tmp_path, local=local
    ).resolve(stage_job(tmp_path), target.state, check_only=False)

    assert target.published == []
    assert result.state is target.state
    assert result.actions == ()


def test_resolve_without_profiles_reads_only_the_local_cache(tmp_path):
    class Untouchable(FakeCheckpointService):
        def inspect(self, job, *, download_root=None):
            raise AssertionError("account checkpoints must not be read")

    target = FakeCheckpointService(
        _state(10, 0), revalidated=_state(10, 2, "acc3/checkpoint")
    )
    local = FakeLocalSource(_state(10, 2))

    result = _service(
        target,
        (("acc1", Untouchable(_state(10, 9))), ("acc3", target)),
        tmp_path,
        local=local,
    ).resolve(
        stage_job(tmp_path), target.state, check_only=False, include_profiles=False
    )

    assert target.published[0][0] is local.state.artifact
    assert result.state.completion.complete == 2
```

- [ ] **Step 2: Write the failing local-cache tests**

Create `seed-pipeline/tests/evaluation/test_rerank_cache_checkpoint.py`:

```python
from pathlib import Path

from tests.integrations.kaggle.factories import rerank_runtime_profile, stage_request

from seed_pipeline.artifacts.manifest import Completion
from seed_pipeline.evaluation.rerank_cache_checkpoint import RerankCacheCheckpoint
from seed_pipeline.integrations.kaggle.artifacts import load_cloud_artifact
from seed_pipeline.integrations.kaggle.models import StageJob, StageName
from seed_pipeline.integrations.kaggle.stages import RerankStage
from seed_pipeline.integrations.kaggle.workers.rerank import run_rerank_worker

MODEL = "qwen3-reranker:0.6b-fp16"


def _job(tmp_path: Path, candidate_bundle) -> StageJob:
    request = stage_request(
        StageName.RERANK,
        MODEL,
        candidate_bundle.data_path,
        output_dir=tmp_path / "remote",
        runtime_profile=rerank_runtime_profile(MODEL),
    )
    return RerankStage().build_job(request)


def _no_scoring(_query: str, _document: str, _model: str) -> float:
    raise AssertionError("pairs from the local cache must not be scored again")


def test_empty_cache_offers_no_checkpoint(tmp_path, candidate_bundle):
    job = _job(tmp_path, candidate_bundle)

    state = RerankCacheCheckpoint(tmp_path / "empty.jsonl").inspect(
        job, download_root=tmp_path / "local"
    )

    assert state.reference is None
    assert state.artifact is None
    assert state.completion == Completion(1, 0, 1)


def test_cached_scores_become_a_checkpoint_the_worker_resumes(
    tmp_path, candidate_bundle, complete_rerank_cache
):
    job = _job(tmp_path, candidate_bundle)

    state = RerankCacheCheckpoint(complete_rerank_cache.path).inspect(
        job, download_root=tmp_path / "local"
    )

    assert state.reference is None
    assert state.completion == Completion(1, 1, 0)
    assert state.artifact is not None
    assert state.artifact.checkpoint_path is not None
    loaded = load_cloud_artifact(
        state.artifact.data_path,
        state.artifact.manifest_path,
        job.identity,
        allow_partial=True,
        expected_artifact_type="rerank_scores",
    )
    assert loaded.strict_identity_match
    config = dict(job.worker_config) | {
        "output_dir": str(tmp_path / "worker"),
        "identity": job.identity.payload,
        "job_sha256": job.identity.sha256,
        "input_files": {
            item.key: {
                "path": str(item.source_path),
                "filename": item.filename,
                "sha256": item.sha256,
            }
            for item in job.input_bundle.files
        },
        "checkpoint_filename": str(state.artifact.checkpoint_path),
    }

    resumed = run_rerank_worker(
        config, score_pair=_no_scoring, emit=lambda _message: None, clock=lambda: 0.0
    )

    assert resumed.completion == Completion(1, 1, 0)
```

The `candidate_bundle` and `complete_rerank_cache` fixtures come from `tests/evaluation/conftest.py`.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest -q tests/integrations/kaggle/test_checkpoint_inheritance.py tests/evaluation/test_rerank_cache_checkpoint.py`
Expected: FAIL — `TypeError: ... unexpected keyword argument 'local'` and `ModuleNotFoundError: No module named 'seed_pipeline.evaluation.rerank_cache_checkpoint'`.

- [ ] **Step 4: Write checkpoints from scored records in the worker module**

In `seed-pipeline/src/seed_pipeline/integrations/kaggle/workers/rerank.py`:
- change `from collections.abc import Callable` to `from collections.abc import Callable, Sequence`;
- change `from seed_pipeline.integrations.kaggle.models import CloudArtifact` to `from seed_pipeline.integrations.kaggle.models import CloudArtifact, JobIdentity`;
- in `run_rerank_worker` replace the journal identity literal `{"reuse_sha256": identity.reuse_sha256},` with `_journal_identity(identity),`;
- add below `_pair_fingerprint`:

```python
def _journal_identity(identity: JobIdentity) -> dict[str, str]:
    return {"reuse_sha256": identity.reuse_sha256}


def write_rerank_checkpoint(
    records: Sequence[dict],
    *,
    identity: JobIdentity,
    total: int,
    output_dir: Path,
) -> CloudArtifact:
    """Seal already-scored pairs as a partial artifact a worker resumes from.

    The journal uses the worker's key, fingerprint and identity, so a kernel that
    mounts this checkpoint reuses every record whose query and document still match.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "rerank_scores.jsonl"
    journal_path = output_dir / "rerank_scores.journal.jsonl"
    journal_path.unlink(missing_ok=True)
    journal = AppendOnlyJournal.open(
        journal_path, _journal_identity(identity), _pair_key, _pair_fingerprint
    )
    journal.append_batch(records)
    complete = journal.compact(records, data_path)
    return artifact_from_output(
        data_path,
        artifact_type="rerank_scores",
        identity=identity,
        total=total,
        complete=complete,
        checkpoint_path=journal_path,
    )
```

- [ ] **Step 5: List the scored records of a candidate set**

In `seed-pipeline/src/seed_pipeline/evaluation/rerank_score_cache.py` replace `validate_subset` with:

```python
def validate_subset(self, candidate_data_path: Path, reranker: str) -> ValidatedSubset:
    expected = self.expected_keys_from_candidates(candidate_data_path, reranker)
    found = self._stored(expected)
    missing = len(expected) - len(found)
    return ValidatedSubset(
        total=len(expected),
        complete=len(found),
        missing=missing,
        sha256=record_subset_sha256(found) if not missing else None,
    )


def available_records(
    self, candidate_data_path: Path, reranker: str
) -> list[dict[str, Any]]:
    """Stored records of the candidate pairs that already have a score."""
    return self._stored(
        self.expected_keys_from_candidates(candidate_data_path, reranker)
    )


def _stored(self, expected: builtins.set[RerankKey]) -> list[dict[str, Any]]:
    return [
        self.record_metadata[key]
        for key in sorted(expected)
        if key in self.record_metadata
    ]
```

- [ ] **Step 6: Create the local cache source**

Create `seed-pipeline/src/seed_pipeline/evaluation/rerank_cache_checkpoint.py`:

```python
"""The local rerank score cache as a checkpoint source for Kaggle sessions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from seed_pipeline.artifacts.manifest import Completion
from seed_pipeline.evaluation.rerank_score_cache import RerankScoreCache
from seed_pipeline.integrations.kaggle.checkpoints import CheckpointState
from seed_pipeline.integrations.kaggle.job_lock import kaggle_cache_lock
from seed_pipeline.integrations.kaggle.models import StageJob
from seed_pipeline.integrations.kaggle.workers.rerank import write_rerank_checkpoint
from seed_pipeline.runtime.catalog import require_model


@dataclass(frozen=True)
class RerankCacheCheckpoint:
    """Offers the scores in `data/cache/rerank_scores/<model>.jsonl` as a checkpoint.

    The cache is the source of truth: it survives account switches, runtime profile
    changes and job identity changes, so it can seed every session.
    """

    cache_path: Path

    def inspect(self, job: StageJob, *, download_root: Path) -> CheckpointState:
        spec = require_model(job.model)
        if spec.rerank_contract is None:
            raise ValueError(f"Reranker {job.model} has no scoring contract")
        candidates = job.input_bundle.file("candidates").source_path
        with kaggle_cache_lock(self.cache_path):
            records = RerankScoreCache(
                self.cache_path,
                model_sha256=spec.sha256,
                request_contract_sha256=spec.rerank_contract.sha256,
            ).available_records(candidates, job.model)
        if not records:
            empty = Completion(job.expected_total, 0, job.expected_total)
            return CheckpointState(None, None, empty)
        artifact = write_rerank_checkpoint(
            records,
            identity=job.identity,
            total=job.expected_total,
            output_dir=download_root,
        )
        return CheckpointState(None, artifact, artifact.completion)
```

- [ ] **Step 7: Add the local source to inheritance**

In `seed-pipeline/src/seed_pipeline/integrations/kaggle/checkpoint_inheritance.py`:
- change the imports to include `from collections.abc import Callable`, `from functools import partial` and `from typing import Protocol`;
- add below the imports:

```python
LOCAL_SOURCE = "local"


class LocalCheckpointSource(Protocol):
    """Progress kept on this machine, offered as a checkpoint without a dataset."""

    def inspect(self, job: StageJob, /, *, download_root: Path) -> CheckpointState: ...
```

- add the keyword parameter `local: LocalCheckpointSource | None = None,` after `temp_root: Path,` in `__init__` and store `self.local = local`;
- replace `resolve` with:

```python
def resolve(
    self,
    job: StageJob,
    target_state: CheckpointState,
    *,
    check_only: bool,
    include_profiles: bool = True,
) -> CheckpointInheritanceResult:
    best = target_state
    best_source = self.target_profile
    with tempfile.TemporaryDirectory(
        prefix="checkpoint-inheritance-", dir=str(self.temp_root)
    ) as raw:
        root = Path(raw)
        sources: list[tuple[str, Callable[[], CheckpointState]]] = []
        if self.local is not None:
            sources.append(
                (
                    LOCAL_SOURCE,
                    partial(self.local.inspect, job, download_root=root / LOCAL_SOURCE),
                )
            )
        if include_profiles:
            sources.extend(
                (
                    candidate.profile_name,
                    partial(
                        candidate.checkpoints.inspect,
                        job,
                        download_root=root / candidate.profile_name,
                    ),
                )
                for candidate in self.candidates
                if candidate.profile_name != self.target_profile
            )
        for name, inspect in sources:
            state = inspect()
            if state.completion.complete > best.completion.complete:
                if state.artifact is None:
                    raise RuntimeError(
                        f"checkpoint candidate {name} has progress without artifact"
                    )
                best = state
                best_source = name

        if best_source == self.target_profile:
            return CheckpointInheritanceResult(target_state, ())
        if check_only:
            return CheckpointInheritanceResult(
                target_state,
                (
                    ReconcileAction(
                        "checkpoint",
                        self.target.reference(job),
                        ActionVerb.SYNC,
                        f"would inherit {best_source} -> {self.target_profile}: "
                        f"{best.completion.complete}/{best.completion.total} pairs",
                    ),
                ),
            )
        assert best.artifact is not None
        self.target.publish_if_better(job, best.artifact, target_state)
        verified = self.target.inspect(job)
        if verified.completion.complete < best.completion.complete:
            raise RuntimeError("mirrored checkpoint verification lost progress")
        return CheckpointInheritanceResult(
            verified,
            (
                ReconcileAction(
                    "checkpoint",
                    verified.reference or self.target.reference(job),
                    ActionVerb.SYNC,
                    f"checkpoint inherited {best_source} -> {self.target_profile}: "
                    f"{best.completion.complete}/{best.completion.total} pairs",
                ),
            ),
        )
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `uv run pytest -q tests/integrations/kaggle/test_checkpoint_inheritance.py tests/evaluation/test_rerank_cache_checkpoint.py tests/evaluation/test_rerank_service.py tests/integrations/kaggle/test_worker_rerank.py`
Expected: PASS.

- [ ] **Step 9: Run the seed gate**

Run the **Seed gate**. Expected: all green.

- [ ] **Step 10: Commit**

```bash
git add seed-pipeline/src/seed_pipeline/integrations/kaggle/workers/rerank.py \
  seed-pipeline/src/seed_pipeline/evaluation/rerank_score_cache.py \
  seed-pipeline/src/seed_pipeline/evaluation/rerank_cache_checkpoint.py \
  seed-pipeline/src/seed_pipeline/integrations/kaggle/checkpoint_inheritance.py \
  seed-pipeline/tests/integrations/kaggle/test_checkpoint_inheritance.py \
  seed-pipeline/tests/evaluation/test_rerank_cache_checkpoint.py
git diff --cached --name-status
git commit -m "feat(seed): offer the local rerank score cache as a checkpoint source

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: The owning profile publishes missing dependency datasets

**Files:**
- Modify: `seed-pipeline/src/seed_pipeline/integrations/kaggle/dependencies.py`
- Modify: `seed-pipeline/src/seed_pipeline/integrations/kaggle/service.py` (`make_orchestrator`)
- Test: `seed-pipeline/tests/integrations/kaggle/test_dependencies.py`, `seed-pipeline/tests/integrations/kaggle/test_service.py`

**Interfaces:**
- Consumes: `KaggleExecutionContext.owners.execution` (profile username), `DatasetService(runner, owner)`, `plan_dataset(desired, observed, *, force)`.
- Produces:
  - `DependencyService(datasets: DatasetService, desired_builder: DesiredBuilder, *, publishers: Sequence[DatasetService] = ())`; attribute `publishers: dict[str, DatasetService]` keyed by casefolded owner, always containing `datasets` itself.
  - `reconcile(...)` creates or updates a dataset `owner/slug` with `publishers[owner]`; when no profile owns the reference it raises `ValueError("No Kaggle account profile can publish <owner>/<slug>; profiles own: ...")` before materializing anything.
  - `default_desired_datasets` puts every input dataset under `owners.corpus or owners.runtime or owners.execution` (the shared owner in profile mode), like the model dataset under `owners.runtime`.
  - `make_orchestrator` passes one `DatasetService(context.runner, context.owners.execution)` per profile context as `publishers`.

Why (spec 4.4, 4.9): running the 4b job with acc3 failed with `Dependency owner doanvanan0209 does not match dataset service owner ...` because the missing model dataset belongs to `KAGGLE_SHARED_OWNER` (acc1).

- [ ] **Step 1: Write the failing tests**

In `seed-pipeline/tests/integrations/kaggle/test_dependencies.py` add these imports:

```python
import pytest
from tests.integrations.kaggle.factories import stage_job

from seed_pipeline.integrations.kaggle.dataset_service import (
    DatasetPresence,
    DatasetRemoteState,
    DatasetService,
    PreparedDataset,
)
from seed_pipeline.integrations.kaggle.models import ActionVerb, StageJob
from seed_pipeline.integrations.kaggle.reconcile import DesiredDataset
```

(merge `stage_job` into the existing `from tests.integrations.kaggle.factories import (...)` block and `ActionVerb`, `StageJob` into the existing models import), change the last assertion block of `test_desired_datasets_multi_owner_resolution` to:

```python
    assert by_kind["model"].reference == "runtime-acc/model-slug"
    assert by_kind["model"].public is True
    assert by_kind["input"].reference.startswith("corpus-acc/pipeline-input-")
    assert by_kind["input"].public is True
```

and append:

```python
class UnusedRunner:
    def run(
        self,
        args: list[str],
        capture_output: bool = False,
        *,
        live_output: bool = False,
    ) -> str:
        raise AssertionError(f"unexpected Kaggle command: {args}")


class FakeKaggleDatasets:
    """Shared remote dataset state seen by every account's DatasetService."""

    def __init__(self, monkeypatch):
        self.monkeypatch = monkeypatch
        self.published: set[str] = set()
        self.created: list[tuple[str, str]] = []

    def service(self, owner: str) -> DatasetService:
        service = DatasetService(UnusedRunner(), owner)

        def inspect_state(reference, *, active_owner=None):
            if reference in self.published:
                return DatasetRemoteState(
                    DatasetPresence.EXISTS, status="READY", current_version=1
                )
            return DatasetRemoteState(DatasetPresence.ABSENT)

        def ensure_dataset(
            slug, title, path, *, public=False, active_owner=None, message=None
        ):
            reference = f"{owner}/{slug}"
            self.created.append((owner, reference))
            self.published.add(reference)
            return PreparedDataset(reference, True, 1)

        def wait_for_dataset_ready(
            reference, *, minimum_version=None, max_attempts=300
        ):
            return None

        self.monkeypatch.setattr(service, "inspect_state", inspect_state)
        self.monkeypatch.setattr(service, "ensure_dataset", ensure_dataset)
        self.monkeypatch.setattr(
            service, "wait_for_dataset_ready", wait_for_dataset_ready
        )
        return service


def _shared_model(job: StageJob, owner_configuration, workspace: Path):
    del job, workspace
    return (
        DesiredDataset(
            "model",
            f"{owner_configuration.runtime}/model-slug",
            "Kaggle Pipeline Model",
            "model-sha",
            "model_manifest.json",
            True,
            lambda root: root,
        ),
    )


def test_missing_shared_dataset_is_created_by_the_shared_owner_profile(
    tmp_path, monkeypatch
):
    kaggle = FakeKaggleDatasets(monkeypatch)
    service = dependencies.DependencyService(
        kaggle.service("tertiary-user"),
        _shared_model,
        publishers=(kaggle.service("primary-user"),),
    )

    actions = service.reconcile(
        stage_job(tmp_path),
        owners("tertiary-user", runtime="primary-user", corpus="primary-user"),
        tmp_path,
        force=False,
        check_only=False,
    )

    assert [action.verb for action in actions] == [ActionVerb.CREATE]
    assert kaggle.created == [("primary-user", "primary-user/model-slug")]


def test_missing_dataset_without_an_owning_profile_fails_before_upload(
    tmp_path, monkeypatch
):
    kaggle = FakeKaggleDatasets(monkeypatch)
    service = dependencies.DependencyService(
        kaggle.service("tertiary-user"), _shared_model
    )

    with pytest.raises(
        ValueError,
        match="No Kaggle account profile can publish primary-user/model-slug",
    ):
        service.reconcile(
            stage_job(tmp_path),
            owners("tertiary-user", runtime="primary-user"),
            tmp_path,
            force=False,
            check_only=False,
        )
    assert kaggle.created == []
```

In `seed-pipeline/tests/integrations/kaggle/test_service.py` add `from seed_pipeline.integrations.kaggle.dependencies import DependencyService` to the imports and append:

```python
def test_make_orchestrator_lets_every_profile_publish_its_own_datasets(
    tmp_path, monkeypatch
):
    env_file = _write_profiles(tmp_path)
    _clear_kaggle_environment(monkeypatch)
    contexts = resolve_profile_execution_contexts(env_file=env_file)
    target = contexts[1]

    orchestrator = kaggle_service.make_orchestrator(
        target.owners,
        target.runner,
        target_profile="acc2",
        checkpoint_contexts=contexts,
        temp_root=tmp_path / "tmp",
    )

    dependency_service = orchestrator.dependencies
    assert isinstance(dependency_service, DependencyService)
    assert sorted(dependency_service.publishers) == ["primary-user", "secondary-user"]
    for owner, publisher in dependency_service.publishers.items():
        runner = publisher.runner
        assert isinstance(runner, KaggleCommandRunner)
        assert runner.environment is not None
        assert runner.environment["KAGGLE_USERNAME"].casefold() == owner
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/integrations/kaggle/test_dependencies.py tests/integrations/kaggle/test_service.py`
Expected: FAIL — `TypeError: DependencyService.__init__() got an unexpected keyword argument 'publishers'`, the input owner assertion (`worker-acc/...`), and `AttributeError: 'DependencyService' object has no attribute 'publishers'`.

- [ ] **Step 3: Publish through the owning profile**

In `seed-pipeline/src/seed_pipeline/integrations/kaggle/dependencies.py` replace the class `DependencyService` with:

```python
class DependencyService:
    def __init__(
        self,
        datasets: DatasetService,
        desired_builder: DesiredBuilder,
        *,
        publishers: Sequence[DatasetService] = (),
    ):
        self.datasets = datasets
        self.desired_builder = desired_builder
        # A dataset can only be created or versioned by the account that owns it, so
        # each reference is published by the profile whose username is its owner.
        self.publishers: dict[str, DatasetService] = {
            service.owner.casefold(): service for service in publishers
        }
        self.publishers[datasets.owner.casefold()] = datasets

    def desired(
        self, job: StageJob, owners: OwnerConfiguration, workspace: Path
    ) -> Sequence[DesiredDataset]:
        return self.desired_builder(job, owners, workspace)

    def require_ready(self, reference: str) -> None:
        self.datasets.require_ready(
            reference,
            guidance="publish the Kaggle runtime dataset before running a stage",
        )

    def reconcile(
        self,
        job: StageJob,
        owners: OwnerConfiguration,
        workspace: Path,
        *,
        force: bool,
        check_only: bool,
    ) -> Sequence[ReconcileAction]:
        desired = tuple(self.desired(job, owners, workspace))
        resources = {item.reference: item.manifest_filename for item in desired}
        inventory = DatasetInventory.load(
            self.datasets, resources, active_owner=owners.execution
        )
        actions: list[ReconcileAction] = []
        for item in desired:
            owner, _, slug = item.reference.partition("/")
            publisher = self.publishers.get(owner.casefold())
            item_force = force and publisher is not None
            action = plan_dataset(item, inventory.get(item.reference), force=item_force)
            if action.verb is ActionVerb.WAIT and not check_only:
                reader = publisher or self.datasets
                reader.wait_for_dataset_ready(item.reference)
                refreshed = reader.inspect_state(
                    item.reference, active_owner=reader.owner
                )
                inventory.remember(item.reference, refreshed)
                action = plan_dataset(item, refreshed, force=item_force)
            if action.verb in {ActionVerb.CREATE, ActionVerb.UPDATE} and not check_only:
                if publisher is None:
                    raise ValueError(
                        f"No Kaggle account profile can publish {item.reference}; "
                        f"profiles own: {', '.join(sorted(self.publishers))}"
                    )
                with tempfile.TemporaryDirectory(
                    prefix=f"dependency-{item.resource_kind}-"
                ) as raw:
                    staged = item.materialize(Path(raw))
                    prepared = publisher.ensure_dataset(
                        slug,
                        item.title,
                        staged,
                        public=item.public,
                        active_owner=publisher.owner,
                    )
                publisher.wait_for_dataset_ready(
                    item.reference, minimum_version=prepared.expected_version
                )
                inventory.remember(
                    item.reference,
                    publisher.inspect_state(
                        item.reference, active_owner=publisher.owner
                    ),
                )
            actions.append(action)
        return actions
```

In `default_desired_datasets` replace

```python
    input_owner = (
        (owners.corpus or owners.runtime or owners.execution)
        if input_slug == BUNDLE_INPUT_DATASET_SLUG
        else owners.execution
    )
```

with

```python
    # Inputs live with the shared owner, like the model, so every account mounts the
    # same dataset and none of them has to create it.
    input_owner = owners.corpus or owners.runtime or owners.execution
```

- [ ] **Step 4: Give the orchestrator one publisher per profile**

In `seed-pipeline/src/seed_pipeline/integrations/kaggle/service.py`, in `make_orchestrator`, replace

```python
    dependencies = DependencyService(
        DatasetService(runner, owners.execution), default_desired_datasets
    )
```

with

```python
    dependencies = DependencyService(
        DatasetService(runner, owners.execution),
        default_desired_datasets,
        publishers=tuple(
            DatasetService(context.runner, context.owners.execution)
            for context in checkpoint_contexts
            if context.profile is not None
        ),
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -q tests/integrations/kaggle/test_dependencies.py tests/integrations/kaggle/test_service.py tests/integrations/kaggle/test_orchestrator.py`
Expected: PASS.

- [ ] **Step 6: Run the seed gate**

Run the **Seed gate**. Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add seed-pipeline/src/seed_pipeline/integrations/kaggle/dependencies.py \
  seed-pipeline/src/seed_pipeline/integrations/kaggle/service.py \
  seed-pipeline/tests/integrations/kaggle/test_dependencies.py \
  seed-pipeline/tests/integrations/kaggle/test_service.py
git diff --cached --name-status
git commit -m "fix(seed): publish missing Kaggle dependencies with the owning profile

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Orchestrator artifact sink and session-sized runs

**Files:**
- Modify: `seed-pipeline/src/seed_pipeline/integrations/kaggle/models.py` (`StageRequest`)
- Modify: `seed-pipeline/src/seed_pipeline/integrations/kaggle/orchestrator.py`
- Modify: `seed-pipeline/src/seed_pipeline/integrations/kaggle/service.py` (`make_orchestrator`, `run_kaggle_stage`)
- Test: `seed-pipeline/tests/integrations/kaggle/test_orchestrator.py`, `seed-pipeline/tests/integrations/kaggle/test_service.py`

**Interfaces:**
- Consumes: `CheckpointInheritanceService.resolve(..., include_profiles)` and `LocalCheckpointSource` (Task 5); `DependencyService(publishers=...)` (Task 6).
- Produces:
  - `StageRequest.resume_remote: bool = True` (last field). `False` makes a run ignore account checkpoints and existing kernels, like `force`, but without republishing dependencies; inheritance still reads the local source.
  - `orchestrator.ArtifactSink = Callable[[StageJob, CloudArtifact], None]`.
  - `CheckpointInheritanceResolver.resolve(self, job, target_state, /, *, check_only: bool, include_profiles: bool = True) -> CheckpointInheritanceResult`.
  - `KagglePipelineOrchestrator(..., artifact_sink: ArtifactSink | None = None)`. The sink receives every non-benchmark artifact that carries results — attached kernel output, submitted kernel output (complete or partial) and a complete checkpoint before promotion — before any checkpoint is published.
  - Re-attaching a finished kernel whose partial output adds no pairs no longer raises `checkpoint publication made no progress`.
  - Attaching a queued or running kernel counts as one of `max_runs`; with `max_runs=1` the run returns after the attach instead of submitting a second kernel.
  - `make_orchestrator(owners, runner, *, target_profile=None, checkpoint_contexts=(), temp_root=Path("/tmp"), local_checkpoint: LocalCheckpointSource | None = None, artifact_sink: ArtifactSink | None = None)`.
  - `run_kaggle_stage(..., resume_remote: bool = True, artifact_sink: ArtifactSink | None = None, local_checkpoint: LocalCheckpointSource | None = None)`; a `local_checkpoint` without an account profile raises `ValueError`.

How a re-run recovers a finished kernel with the same job identity: `run()` inspects the account's kernel, `attach_or_recover` downloads its output, the artifact sink merges the partial scores into the local cache, and `publish_if_better` publishes a checkpoint only if it adds pairs. After Plan A the rerank job identity of the 15/09 kernel differs (new runtime profile candidates, new `reuse_sha256`), so its kernel and checkpoint slugs are no longer found; Task 10 adds `--recover-kernel` for that case, and from then on the local cache carries the progress.

- [ ] **Step 1: Update the orchestrator test fakes**

In `seed-pipeline/tests/integrations/kaggle/test_orchestrator.py`:

Replace `FakeDependencies` and `FakeCheckpoints` with:

```python
class FakeDependencies:
    def __init__(self):
        self.required = 0
        self.reconciles = 0
        self.forced: list[bool] = []

    def require_ready(self, _reference):
        self.required += 1

    def reconcile(self, *_args, **kwargs):
        self.reconciles += 1
        self.forced.append(kwargs["force"])
        return []


class FakeCheckpoints:
    def __init__(self, total, state=None):
        self.empty_state = CheckpointState(None, None, Completion(total, 0, total))
        self.current = state or self.empty_state
        self.inspected = []
        self.emptied = []
        self.published = []

    def empty(self, job):
        self.emptied.append(job)
        return self.empty_state

    def inspect(self, job):
        self.inspected.append(job)
        return self.current

    def publish_if_better(self, _job, artifact, current):
        if artifact.completion.complete <= current.completion.complete:
            return current
        self.published.append(artifact.completion)
        return CheckpointState("owner/checkpoint", artifact, artifact.completion)

    def reference(self, _job):
        return "owner/checkpoint"
```

Replace `FakeInheritance.resolve` with:

```python
    def resolve(
        self,
        job: StageJob,
        state: CheckpointState,
        *,
        check_only: bool,
        include_profiles: bool = True,
    ) -> CheckpointInheritanceResult:
        self.calls.append((job, state, check_only, include_profiles))
        return CheckpointInheritanceResult(self.state, self.actions)
```

In `test_inheritance_failure_prevents_kernel_submission` replace the signature of `BrokenInheritance.resolve` with:

```python
        def resolve(
            self,
            job: StageJob,
            target_state: CheckpointState,
            *,
            check_only: bool,
            include_profiles: bool = True,
        ) -> CheckpointInheritanceResult:
```

Replace `_request` with:

```python
def _request(tmp_path, *, candidates: int = 1):
    candidates_path = tmp_path / "candidates.jsonl"
    candidates_path.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "query": "query",
                "candidates": [
                    {"chunk_id": f"c{index}", "document_text": "text"}
                    for index in range(1, candidates + 1)
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    return stage_request(
        StageName.RERANK,
        "qwen3-reranker:0.6b-fp16",
        candidates_path,
        output_dir=tmp_path / "remote",
        owner_configuration=owners(),
        max_runs=1,
        total_budget_seconds=60,
        runtime_profile=rerank_runtime_profile(),
    )
```

- [ ] **Step 2: Write the failing orchestrator tests**

Append to `seed-pipeline/tests/integrations/kaggle/test_orchestrator.py`:

```python
def _artifact(tmp_path, job, *, complete, total, name="output"):
    data = tmp_path / name / "rerank_scores.jsonl"
    data.parent.mkdir(parents=True)
    data.write_text(
        "".join(json.dumps({"pair": index}) + "\n" for index in range(complete)),
        encoding="utf-8",
    )
    return artifact_from_output(
        data,
        artifact_type="rerank_scores",
        identity=job.identity,
        total=total,
        complete=complete,
    )


def _detached_submission():
    queued = KernelRemoteState(
        "owner/kernel", KernelPresence.EXISTS, KernelStatus.QUEUED
    )
    return KernelResolution(queued, None, (), submitted=True)


def test_reattached_output_without_new_pairs_does_not_fail(tmp_path, monkeypatch):
    request = _request(tmp_path, candidates=2)
    job = RerankStage().build_job(request)
    artifact = _artifact(tmp_path, job, complete=1, total=2)
    checkpoints = FakeCheckpoints(
        2, state=CheckpointState("owner/checkpoint", artifact, Completion(2, 1, 1))
    )
    finished = KernelRemoteState(
        "owner/kernel", KernelPresence.EXISTS, KernelStatus.COMPLETE
    )
    reconciler = FakeReconciler(
        KernelResolution(finished, artifact.data_path.parent, ()),
        _detached_submission(),
    )
    delivered = []
    orchestrator = KagglePipelineOrchestrator(
        RerankStage(),
        FakeDependencies(),
        checkpoints,
        FakeKernels(),
        reconciler=reconciler,
        artifact_sink=lambda _job, item: delivered.append(item.completion),
    )
    monkeypatch.setattr(
        orchestrator, "_load_downloaded_artifact", lambda root, job: artifact
    )

    result = orchestrator.run(request)

    assert checkpoints.published == []
    assert delivered == [Completion(2, 1, 1)]
    assert len(reconciler.submissions) == 1
    assert result.completion == Completion(2, 1, 1)


def test_session_output_reaches_the_sink_before_checkpoint_publication(
    tmp_path, monkeypatch
):
    request = _request(tmp_path, candidates=2)
    job = RerankStage().build_job(request)
    artifact = _artifact(tmp_path, job, complete=1, total=2)
    events: list[str] = []

    class OrderedCheckpoints(FakeCheckpoints):
        def publish_if_better(self, _job, artifact, current):
            events.append("publish")
            return super().publish_if_better(_job, artifact, current)

    absent = KernelRemoteState("owner/kernel", KernelPresence.ABSENT)
    done = KernelRemoteState(
        "owner/kernel", KernelPresence.EXISTS, KernelStatus.COMPLETE
    )
    reconciler = FakeReconciler(
        KernelResolution(absent, None, ()),
        KernelResolution(done, artifact.data_path.parent, (), submitted=True),
    )
    orchestrator = KagglePipelineOrchestrator(
        RerankStage(),
        FakeDependencies(),
        OrderedCheckpoints(2),
        FakeKernels(),
        reconciler=reconciler,
        artifact_sink=lambda _job, _artifact: events.append("sink"),
    )
    monkeypatch.setattr(
        orchestrator, "_load_downloaded_artifact", lambda root, job: artifact
    )

    result = orchestrator.run(request)

    assert events == ["sink", "publish"]
    assert result.completion == Completion(2, 1, 1)


def test_attaching_a_running_kernel_uses_up_a_one_run_request(tmp_path, monkeypatch):
    request = _request(tmp_path, candidates=2)
    job = RerankStage().build_job(request)
    artifact = _artifact(tmp_path, job, complete=1, total=2)
    running = KernelRemoteState(
        "owner/kernel", KernelPresence.EXISTS, KernelStatus.RUNNING
    )
    reconciler = FakeReconciler(
        KernelResolution(running, artifact.data_path.parent, ()),
        _detached_submission(),
    )
    checkpoints = FakeCheckpoints(2)
    dependencies = FakeDependencies()
    orchestrator = KagglePipelineOrchestrator(
        RerankStage(), dependencies, checkpoints, FakeKernels(), reconciler=reconciler
    )
    monkeypatch.setattr(
        orchestrator, "_load_downloaded_artifact", lambda root, job: artifact
    )

    result = orchestrator.run(request)

    assert reconciler.submissions == []
    assert dependencies.reconciles == 0
    assert checkpoints.published == [Completion(2, 1, 1)]
    assert result.run_count == 1
    assert result.completion == Completion(2, 1, 1)


def test_run_without_remote_resume_ignores_account_state(tmp_path):
    request = replace(_request(tmp_path), resume_remote=False)
    checkpoints = FakeCheckpoints(1)
    inheritance = FakeInheritance(CheckpointState(None, None, Completion(1, 0, 1)))
    finished = KernelRemoteState(
        "owner/kernel", KernelPresence.EXISTS, KernelStatus.COMPLETE
    )
    reconciler = FakeReconciler(
        KernelResolution(finished, None, ()), _detached_submission()
    )
    dependencies = FakeDependencies()
    orchestrator = KagglePipelineOrchestrator(
        RerankStage(),
        dependencies,
        checkpoints,
        FakeKernels(),
        reconciler=reconciler,
        checkpoint_inheritance=inheritance,
    )

    orchestrator.run(request)

    assert checkpoints.inspected == []
    assert inheritance.calls[0][3] is False
    assert reconciler.attached == []
    assert len(reconciler.submissions) == 1
    assert dependencies.forced == [False]
```

- [ ] **Step 3: Write the failing service tests**

In `seed-pipeline/tests/integrations/kaggle/test_service.py` add to the imports:

```python
from seed_pipeline.integrations.kaggle.checkpoints import CheckpointState
from seed_pipeline.integrations.kaggle.models import CloudArtifact, StageJob
```

(merge `CloudArtifact`, `StageJob` into the existing `models` import) and append:

```python
class EmptyLocalSource:
    def inspect(self, job: StageJob, *, download_root: Path) -> CheckpointState:
        raise AssertionError("not used by this test")


def _ignore_artifact(job: StageJob, artifact: CloudArtifact) -> None:
    del job, artifact


def test_run_stage_passes_session_hooks_to_the_orchestrator(tmp_path, monkeypatch):
    env_file = _write_profiles(tmp_path)
    _clear_kaggle_environment(monkeypatch)
    captured = {}
    source = EmptyLocalSource()

    class FakeOrchestrator:
        def run(self, request):
            captured["request"] = request
            return "result"

    def fake_make_orchestrator(owners, runner, **kwargs):
        captured["kwargs"] = kwargs
        return FakeOrchestrator()

    monkeypatch.setattr(kaggle_service, "make_orchestrator", fake_make_orchestrator)

    kaggle_service.run_kaggle_stage(
        stage=StageName.RERANK,
        model=MODEL,
        input_path=tmp_path / "candidates.jsonl",
        output_dir=tmp_path / "output",
        runtime_profile=rerank_runtime_profile(MODEL),
        kaggle_account="acc2",
        env_file=env_file,
        max_runs=1,
        resume_remote=False,
        artifact_sink=_ignore_artifact,
        local_checkpoint=source,
    )

    assert captured["request"].max_runs == 1
    assert captured["request"].resume_remote is False
    assert captured["kwargs"]["artifact_sink"] is _ignore_artifact
    assert captured["kwargs"]["local_checkpoint"] is source


def test_make_orchestrator_wires_the_local_source_and_sink(tmp_path, monkeypatch):
    env_file = _write_profiles(tmp_path)
    _clear_kaggle_environment(monkeypatch)
    contexts = resolve_profile_execution_contexts(env_file=env_file)
    source = EmptyLocalSource()

    orchestrator = kaggle_service.make_orchestrator(
        contexts[0].owners,
        contexts[0].runner,
        target_profile="acc1",
        checkpoint_contexts=contexts,
        temp_root=tmp_path / "tmp",
        local_checkpoint=source,
        artifact_sink=_ignore_artifact,
    )

    inheritance = orchestrator.checkpoint_inheritance
    assert isinstance(inheritance, CheckpointInheritanceService)
    assert inheritance.local is source
    assert orchestrator.artifact_sink is _ignore_artifact
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `uv run pytest -q tests/integrations/kaggle/test_orchestrator.py tests/integrations/kaggle/test_service.py`
Expected: FAIL — `TypeError: ... unexpected keyword argument 'artifact_sink'`, `TypeError: ... 'resume_remote'`, and `RuntimeError: checkpoint publication made no progress` in `test_reattached_output_without_new_pairs_does_not_fail`.

- [ ] **Step 5: Add `resume_remote` to `StageRequest`**

In `seed-pipeline/src/seed_pipeline/integrations/kaggle/models.py` add as the last field of `StageRequest`, after `runtime_profile: RuntimeCandidate | None = None`:

```python
    # False: ignore account checkpoints and existing kernels (a forced command's
    # later sessions) without republishing dependencies.
    resume_remote: bool = True
```

- [ ] **Step 6: Implement the orchestrator changes**

In `seed-pipeline/src/seed_pipeline/integrations/kaggle/orchestrator.py`:

Change the imports: add `from collections.abc import Callable`; add `CloudArtifact` to the `models` import.

Below `RUNTIME_DATASET_SLUG = ...` add:

```python
ArtifactSink = Callable[[StageJob, CloudArtifact], None]
```

Replace `CheckpointInheritanceResolver` with:

```python
class CheckpointInheritanceResolver(Protocol):
    """Resolves the best checkpoint across the local source and account profiles."""

    def resolve(
        self,
        job: StageJob,
        target_state: CheckpointState,
        /,
        *,
        check_only: bool,
        include_profiles: bool = True,
    ) -> CheckpointInheritanceResult: ...
```

In `KagglePipelineOrchestrator.__init__` add the keyword parameter `artifact_sink: ArtifactSink | None = None,` after `checkpoint_inheritance` and store `self.artifact_sink = artifact_sink`.

Replace `run` and `_run_until_complete` with:

```python
def run(self, request: StageRequest) -> PipelineResult:
    if request.max_runs < 1:
        raise ValueError("max_runs must be at least 1")
    job = self._adapter(request.stage).build_job(request)
    benchmark = job.stage.value.endswith("-benchmark")
    local = self._inspect_local(job)
    if not request.force and local is not None and local.completion.is_complete:
        return PipelineResult(job, local.completion, (), local.data_path, 0)

    ignore_remote = request.force or not request.resume_remote
    checkpoint_actions: tuple[ReconcileAction, ...] = ()
    if benchmark or ignore_remote:
        checkpoint = self.checkpoints.empty(job)
    else:
        checkpoint = self.checkpoints.inspect(job)
    if not benchmark and self.checkpoint_inheritance is not None:
        inheritance = self.checkpoint_inheritance.resolve(
            job,
            checkpoint,
            check_only=request.check_only,
            include_profiles=not ignore_remote,
        )
        checkpoint = inheritance.state
        checkpoint_actions = inheritance.actions
    remote = (
        KernelRemoteState(self.reconciler.reference(job), KernelPresence.ABSENT)
        if ignore_remote
        else self.reconciler.inspect(job)
    )

    if request.check_only:
        return PipelineResult(
            job,
            checkpoint.completion,
            (*checkpoint_actions, *self._check_actions(job, remote, checkpoint)),
            None,
            0,
        )

    with managed_staging_directory(
        self.temp_root, prefix="kaggle-pipeline-"
    ) as workspace:
        actions: tuple[ReconcileAction, ...] = checkpoint_actions
        # A queued or running kernel is a session already in progress.
        attached_runs = int(
            remote.presence is KernelPresence.EXISTS
            and remote.status in {KernelStatus.QUEUED, KernelStatus.RUNNING}
        )

        if remote.presence is KernelPresence.EXISTS:
            resolution = self.reconciler.attach_or_recover(
                job,
                remote,
                workspace,
                timeout_seconds=request.total_budget_seconds,
            )
            actions += resolution.actions
            if resolution.output_root is not None:
                artifact = None
                try:
                    artifact = self._load_downloaded_artifact(
                        resolution.output_root, job
                    )
                except ArtifactContractError as error:
                    if (
                        resolution.remote.status is not KernelStatus.ERROR
                        or resolution.submitted
                    ):
                        raise ArtifactContractError(
                            f"{error}; kernel={resolution.remote.reference}; "
                            f"log_tail={self.reconciler.log_tail(resolution.remote.reference)}"
                        ) from error
                if artifact is not None:
                    if not benchmark:
                        self._deliver(job, artifact)
                    if artifact.completion.is_complete:
                        return self._complete_result(job, artifact, actions, attempts=0)
                    if not benchmark:
                        # A re-attached kernel may hold pairs the checkpoint
                        # already has; that is not an error.
                        published = self.checkpoints.publish_if_better(
                            job, artifact, checkpoint
                        )
                        if (
                            published.completion.complete
                            > checkpoint.completion.complete
                        ):
                            checkpoint = published
                            actions += (self._checkpoint_action(checkpoint),)

        if (
            remote.presence is KernelPresence.ABSENT
            and checkpoint.completion.is_complete
            and checkpoint.artifact is not None
            and checkpoint.artifact.strict_identity_match
        ):
            if not benchmark:
                self._deliver(job, checkpoint.artifact)
            destination = promote_complete_artifact(
                checkpoint.artifact, job.local_cache_path
            )
            return PipelineResult(
                job,
                checkpoint.completion,
                (*actions, self._finalize_action(destination, "checkpoint")),
                destination,
                0,
            )

        runs = request.max_runs - attached_runs
        if runs < 1:
            return PipelineResult(
                job, checkpoint.completion, actions, None, attached_runs
            )
        return self._run_until_complete(
            job, request, workspace, actions, checkpoint, runs=runs
        )


def _run_until_complete(
    self,
    job,
    request,
    workspace,
    actions: tuple[ReconcileAction, ...],
    checkpoint: CheckpointState,
    *,
    runs: int,
) -> PipelineResult:
    benchmark = job.stage.value.endswith("-benchmark")
    state = checkpoint
    for attempt in range(1, runs + 1):
        self.dependencies.require_ready(runtime_dataset_reference(request.owners))
        dependency_actions = tuple(
            self.dependencies.reconcile(
                job,
                request.owners,
                workspace,
                force=request.force,
                check_only=request.check_only,
            )
        )
        dataset_references = [
            runtime_dataset_reference(request.owners),
            *(
                action.reference
                for action in dependency_actions
                if action.resource_kind != "artifact"
            ),
        ]
        bundle = self.kernels.prepare_bundle(
            job,
            root=workspace / str(attempt),
            dataset_references=dataset_references,
            checkpoint_reference=None if benchmark else state.reference,
            total_budget_seconds=request.total_budget_seconds,
        )
        resolution = self.reconciler.submit(
            job,
            bundle,
            workspace / str(attempt),
            timeout_seconds=request.total_budget_seconds,
        )
        if resolution.output_root is None:
            return PipelineResult(
                job,
                state.completion,
                (*actions, *dependency_actions, *resolution.actions),
                None,
                attempt,
            )
        try:
            artifact = self._load_downloaded_artifact(resolution.output_root, job)
        except ArtifactContractError as error:
            raise ArtifactContractError(
                f"{error}; kernel={resolution.remote.reference}; "
                f"log_tail={self.reconciler.log_tail(resolution.remote.reference)}"
            ) from error
        combined_actions = (*actions, *dependency_actions, *resolution.actions)
        if not benchmark:
            self._deliver(job, artifact)
        if artifact.completion.is_complete:
            return self._complete_result(
                job, artifact, combined_actions, attempts=attempt
            )
        if benchmark:
            actions = combined_actions
        else:
            state = self._publish_checkpoint(job, artifact, state)
            actions = (*combined_actions, self._checkpoint_action(state))
    return PipelineResult(job, state.completion, actions, None, runs)


def _deliver(self, job: StageJob, artifact: CloudArtifact) -> None:
    if self.artifact_sink is not None:
        self.artifact_sink(job, artifact)
```

If Plan A changed lines inside these two methods (for example the `prepare_bundle` arguments), keep Plan A's version of those lines and apply only the changes described in the Interfaces block.

- [ ] **Step 7: Pass the hooks through the service**

In `seed-pipeline/src/seed_pipeline/integrations/kaggle/service.py`:

Add to the imports:

```python
from seed_pipeline.integrations.kaggle.checkpoint_inheritance import (
    CheckpointInheritanceService,
    LocalCheckpointSource,
    ProfileCheckpointService,
)
from seed_pipeline.integrations.kaggle.orchestrator import (
    ArtifactSink,
    KagglePipelineOrchestrator,
)
```

(replacing the existing imports of those two modules).

In `make_orchestrator` add the keyword parameters `local_checkpoint: LocalCheckpointSource | None = None,` and `artifact_sink: ArtifactSink | None = None,` after `temp_root`, add `local=local_checkpoint,` to the `CheckpointInheritanceService(...)` call, and add `artifact_sink=artifact_sink,` to the `KagglePipelineOrchestrator(...)` call.

Replace `run_kaggle_stage` with:

```python
def run_kaggle_stage(
    *,
    stage: StageName,
    model: str,
    input_path: Path,
    output_dir: Path,
    gguf_root: Path = GGUF_ROOT,
    force: bool = False,
    check_only: bool = False,
    max_runs: int = 10,
    budget_seconds: int = 21_600,
    benchmark_items: int | None = None,
    runtime_profile: RuntimeCandidate | None = None,
    env_file: Path = DEFAULT_ENV_PATH,
    kaggle_account: str | None = None,
    resume_remote: bool = True,
    artifact_sink: ArtifactSink | None = None,
    local_checkpoint: LocalCheckpointSource | None = None,
) -> PipelineResult:
    context = resolve_execution_context(kaggle_account, env_file=env_file)
    if local_checkpoint is not None and context.profile is None:
        raise ValueError("a local checkpoint source needs a Kaggle account profile")
    runner = context.runner
    owners = context.owners
    request = StageRequest(
        stage,
        model,
        Path(input_path),
        Path(output_dir),
        Path(gguf_root),
        owners,
        force,
        check_only,
        max_runs,
        budget_seconds,
        benchmark_items,
        runtime_profile,
        resume_remote,
    )
    contexts = (
        resolve_profile_execution_contexts(env_file=env_file)
        if context.profile is not None
        else ()
    )
    with unwind_on_sigterm():
        return make_orchestrator(
            owners,
            runner,
            target_profile=context.profile.name if context.profile else None,
            checkpoint_contexts=contexts,
            local_checkpoint=local_checkpoint,
            artifact_sink=artifact_sink,
        ).run(request)
```

If Plan A added parameters to `run_kaggle_stage`, keep them and add the three new ones after them.

- [ ] **Step 8: Run the tests to verify they pass**

Run: `uv run pytest -q tests/integrations/kaggle/`
Expected: PASS.

- [ ] **Step 9: Run the seed gate**

Run the **Seed gate**. Expected: all green.

- [ ] **Step 10: Commit**

```bash
git add seed-pipeline/src/seed_pipeline/integrations/kaggle/models.py \
  seed-pipeline/src/seed_pipeline/integrations/kaggle/orchestrator.py \
  seed-pipeline/src/seed_pipeline/integrations/kaggle/service.py \
  seed-pipeline/tests/integrations/kaggle/test_orchestrator.py \
  seed-pipeline/tests/integrations/kaggle/test_service.py
git diff --cached --name-status
git commit -m "feat(seed): hand every Kaggle artifact to a sink before checkpointing

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Quota-aware session runner

**Files:**
- Create: `seed-pipeline/src/seed_pipeline/integrations/kaggle/sessions.py`
- Modify: `seed-pipeline/src/seed_pipeline/integrations/kaggle/service.py` (add `AUTO_ACCOUNT`, `resolve_session_contexts`, `active_kernel_profile`)
- Test: `seed-pipeline/tests/integrations/kaggle/test_sessions.py` (new), `seed-pipeline/tests/integrations/kaggle/test_service.py`

**Interfaces:**
- Consumes: `read_account_quota`, `select_session_account`, `quota_table`, `MINIMUM_SESSION_SECONDS` (Task 2); `kaggle_account_lock`, `KaggleAccountBusy` (Task 3); `PipelineResult(job, completion, actions, artifact_path, run_count)`.
- Produces (`seed_pipeline.integrations.kaggle.service`):
  - `AUTO_ACCOUNT = "auto"`
  - `resolve_session_contexts(kaggle_account: str | None, *, env_file: Path = DEFAULT_ENV_PATH) -> tuple[KaggleExecutionContext, ...]` — every profile for `auto`, otherwise the one selected (`None` = `KAGGLE_ACCOUNT_DEFAULT`); raises `ValueError("Kaggle sessions need account profiles: ...")` without profiles.
  - `active_kernel_profile(contexts: Sequence[KaggleExecutionContext], *, stage: StageName, model: str, input_path: Path, output_dir: Path, gguf_root: Path = GGUF_ROOT, runtime_profile: RuntimeCandidate | None = None) -> str | None` — the profile whose account has this job's kernel queued or running.
- Produces (`seed_pipeline.integrations.kaggle.sessions`):
  - `class QuotaExhausted(RuntimeError)` with `quotas: tuple[AccountQuota, ...]`, `locked: frozenset[str]`, property `table -> tuple[str, ...]`.
  - `@dataclass(frozen=True) ReservedAccount(context: KaggleExecutionContext, quota: AccountQuota, budget_seconds: int)` with property `profile -> str`.
  - `class StopReason(StrEnum)`: `COMPLETE="complete"`, `QUOTA_EXHAUSTED="quota-exhausted"`, `MAX_RUNS="max-runs"`, `NO_PROGRESS="no-progress"`, `CHECK_ONLY="check-only"`.
  - `@dataclass(frozen=True) SessionRunResult(result: PipelineResult | None, sessions: int, stop_reason: StopReason, quota_table: tuple[str, ...] = ())`.
  - `reserve_session_account(contexts, *, requested_budget_seconds: int, log: Callable[[str], None], read_quota: Callable[[KaggleExecutionContext], AccountQuota] | None = None, account_lock: Callable[[str], AbstractContextManager[str]] | None = None, preferred_profile: str | None = None) -> AbstractContextManager[ReservedAccount]` — reads every quota, locks the preferred profile if given and free (budget = requested), otherwise the best free account; raises `QuotaExhausted`.
  - `run_account_sessions(contexts, run_session: Callable[[ReservedAccount, int], PipelineResult], *, requested_budget_seconds: int, max_sessions: int | None, check_only: bool, log: Callable[[str], None], read_quota=None, account_lock=None, preferred_profile: str | None = None) -> SessionRunResult` — `run_session` receives the reservation and the 1-based session number; the preferred profile applies to session 1 only. It logs each account's quota, the chosen account and budget, every action, the pairs after the session, the chosen account's quota after the session, errors and the stop reason with the quota table. It stops when the job completes, no account has a 1 h budget, a session adds no pairs compared with the previous session, `max_sessions` is reached, or after one session in `check_only` mode. `requested_budget_seconds < 3600` or `max_sessions < 1` raise `ValueError`.

- [ ] **Step 1: Write the failing session tests**

Create `seed-pipeline/tests/integrations/kaggle/test_sessions.py`:

```python
from datetime import datetime
from pathlib import Path

import pytest
from tests.integrations.kaggle.factories import owners, stage_job

from seed_pipeline.artifacts.manifest import Completion
from seed_pipeline.integrations.kaggle.api import KaggleCommandRunner
from seed_pipeline.integrations.kaggle.config import KaggleAccountProfile
from seed_pipeline.integrations.kaggle.job_lock import (
    KaggleAccountBusy,
    kaggle_account_lock,
)
from seed_pipeline.integrations.kaggle.models import PipelineResult
from seed_pipeline.integrations.kaggle.quota import AccountQuota
from seed_pipeline.integrations.kaggle.service import KaggleExecutionContext
from seed_pipeline.integrations.kaggle.sessions import (
    ReservedAccount,
    StopReason,
    run_account_sessions,
)

REFRESH = datetime(2026, 9, 19)


def _contexts(*names: str) -> tuple[KaggleExecutionContext, ...]:
    return tuple(
        KaggleExecutionContext(
            KaggleAccountProfile(name, f"user-{name}", f"token-{name}"),
            owners(f"user-{name}"),
            KaggleCommandRunner(environment={}),
        )
        for name in names
    )


class QuotaBook:
    """Fake `kaggle quota -v` per account; sessions spend hours from it."""

    def __init__(self, **remaining: float):
        self.remaining = dict(remaining)
        self.reads: list[str] = []

    def __call__(self, context: KaggleExecutionContext) -> AccountQuota:
        assert context.profile is not None
        name = context.profile.name
        self.reads.append(name)
        return AccountQuota(
            name, context.profile.username, self.remaining[name], 30.0, REFRESH
        )


class ScriptedSessions:
    def __init__(
        self,
        tmp_path: Path,
        book: QuotaBook,
        completes: list[int],
        *,
        total: int = 300_000,
        spent_hours: float = 6.0,
    ):
        self.job = stage_job(tmp_path)
        self.book = book
        self.completes = completes
        self.total = total
        self.spent_hours = spent_hours
        self.accounts: list[tuple[str, int]] = []

    def __call__(self, reserved: ReservedAccount, index: int) -> PipelineResult:
        self.accounts.append((reserved.profile, reserved.budget_seconds))
        self.book.remaining[reserved.profile] -= self.spent_hours
        complete = self.completes[index - 1]
        completion = Completion(self.total, complete, self.total - complete)
        return PipelineResult(self.job, completion, (), None, 1)


def _quiet(_line: str) -> None:
    return None


def test_sessions_move_to_the_account_with_the_most_quota_until_complete(tmp_path):
    book = QuotaBook(acc1=27.58, acc2=29.61, acc3=30.0)
    sessions = ScriptedSessions(tmp_path, book, [100_000, 200_000, 300_000])
    log: list[str] = []

    outcome = run_account_sessions(
        _contexts("acc1", "acc2", "acc3"),
        sessions,
        requested_budget_seconds=21_600,
        max_sessions=None,
        check_only=False,
        log=log.append,
        read_quota=book,
    )

    assert outcome.stop_reason is StopReason.COMPLETE
    assert outcome.sessions == 3
    assert sessions.accounts == [
        ("acc3", 21_600),
        ("acc2", 21_600),
        ("acc1", 21_600),
    ]
    assert any("quota account=acc1 remaining=27.58h" in line for line in log)
    assert any("session=1 start account=acc3" in line for line in log)
    assert any(
        "session=1 quota_after account=acc3 remaining=24.00h" in line for line in log
    )
    assert any("session=3 end pairs=300000/300000" in line for line in log)


def test_session_budget_is_remaining_quota_minus_half_an_hour(tmp_path):
    book = QuotaBook(acc1=4.0)
    sessions = ScriptedSessions(tmp_path, book, [300_000])

    run_account_sessions(
        _contexts("acc1"),
        sessions,
        requested_budget_seconds=21_600,
        max_sessions=None,
        check_only=False,
        log=_quiet,
        read_quota=book,
    )

    assert sessions.accounts == [("acc1", 4 * 3600 - 1800)]


def test_sessions_stop_and_report_refresh_times_when_quota_runs_out(tmp_path):
    book = QuotaBook(acc1=7.0, acc2=1.2)
    sessions = ScriptedSessions(tmp_path, book, [100_000, 150_000])
    log: list[str] = []

    outcome = run_account_sessions(
        _contexts("acc1", "acc2"),
        sessions,
        requested_budget_seconds=21_600,
        max_sessions=None,
        check_only=False,
        log=log.append,
        read_quota=book,
    )

    assert outcome.stop_reason is StopReason.QUOTA_EXHAUSTED
    assert outcome.sessions == 1
    assert outcome.result is not None
    assert outcome.result.completion.complete == 100_000
    assert len(outcome.quota_table) == 3
    assert all("2026-09-19T00:00:00" in row for row in outcome.quota_table[1:])
    assert any("stop=quota-exhausted" in line for line in log)
    assert any("2026-09-19T00:00:00" in line for line in log)


def test_no_session_starts_without_an_hour_of_quota(tmp_path):
    book = QuotaBook(acc1=1.2)
    sessions = ScriptedSessions(tmp_path, book, [1])

    outcome = run_account_sessions(
        _contexts("acc1"),
        sessions,
        requested_budget_seconds=21_600,
        max_sessions=None,
        check_only=False,
        log=_quiet,
        read_quota=book,
    )

    assert outcome.stop_reason is StopReason.QUOTA_EXHAUSTED
    assert outcome.result is None
    assert sessions.accounts == []


def test_sessions_skip_an_account_held_by_another_job(tmp_path):
    book = QuotaBook(acc1=27.58, acc3=30.0)
    sessions = ScriptedSessions(tmp_path, book, [10])
    log: list[str] = []

    with kaggle_account_lock("acc3"):
        run_account_sessions(
            _contexts("acc1", "acc3"),
            sessions,
            requested_budget_seconds=21_600,
            max_sessions=1,
            check_only=False,
            log=log.append,
            read_quota=book,
        )

    assert [profile for profile, _budget in sessions.accounts] == ["acc1"]
    assert any("account=acc3 is locked by another local job" in line for line in log)


def test_the_account_stays_locked_for_the_whole_session(tmp_path):
    book = QuotaBook(acc1=30.0)
    job = stage_job(tmp_path)
    seen: list[str] = []

    def run_session(reserved: ReservedAccount, index: int) -> PipelineResult:
        with (
            pytest.raises(KaggleAccountBusy),
            kaggle_account_lock(reserved.profile),
        ):
            raise AssertionError("the account lock must be held during the session")
        seen.append(reserved.profile)
        return PipelineResult(job, Completion(1, 1, 0), (), None, 1)

    run_account_sessions(
        _contexts("acc1"),
        run_session,
        requested_budget_seconds=21_600,
        max_sessions=None,
        check_only=False,
        log=_quiet,
        read_quota=book,
    )

    assert seen == ["acc1"]
    with kaggle_account_lock("acc1"):
        pass


def test_first_session_attaches_to_the_account_running_the_job(tmp_path):
    book = QuotaBook(acc1=30.0, acc2=0.2)
    sessions = ScriptedSessions(tmp_path, book, [100_000, 300_000])
    log: list[str] = []

    run_account_sessions(
        _contexts("acc1", "acc2"),
        sessions,
        requested_budget_seconds=21_600,
        max_sessions=None,
        check_only=False,
        log=log.append,
        read_quota=book,
        preferred_profile="acc2",
    )

    assert sessions.accounts == [("acc2", 21_600), ("acc1", 21_600)]
    assert any("account=acc2 already runs this job's kernel" in line for line in log)


def test_sessions_stop_when_a_session_adds_no_pairs(tmp_path):
    book = QuotaBook(acc1=30.0)
    sessions = ScriptedSessions(tmp_path, book, [5, 5, 9], spent_hours=1.0)

    outcome = run_account_sessions(
        _contexts("acc1"),
        sessions,
        requested_budget_seconds=21_600,
        max_sessions=None,
        check_only=False,
        log=_quiet,
        read_quota=book,
    )

    assert outcome.stop_reason is StopReason.NO_PROGRESS
    assert outcome.sessions == 2


def test_max_sessions_limits_the_loop(tmp_path):
    book = QuotaBook(acc1=30.0)
    sessions = ScriptedSessions(tmp_path, book, [1, 2, 3], spent_hours=1.0)

    outcome = run_account_sessions(
        _contexts("acc1"),
        sessions,
        requested_budget_seconds=21_600,
        max_sessions=2,
        check_only=False,
        log=_quiet,
        read_quota=book,
    )

    assert outcome.stop_reason is StopReason.MAX_RUNS
    assert outcome.sessions == 2


def test_check_only_plans_one_session_without_reading_quota_afterwards(tmp_path):
    book = QuotaBook(acc1=30.0, acc2=10.0)
    sessions = ScriptedSessions(tmp_path, book, [0], spent_hours=0.0)

    outcome = run_account_sessions(
        _contexts("acc1", "acc2"),
        sessions,
        requested_budget_seconds=21_600,
        max_sessions=None,
        check_only=True,
        log=_quiet,
        read_quota=book,
    )

    assert outcome.stop_reason is StopReason.CHECK_ONLY
    assert book.reads == ["acc1", "acc2"]


def test_a_failing_session_is_logged_and_raised(tmp_path):
    log: list[str] = []

    def broken(reserved: ReservedAccount, index: int) -> PipelineResult:
        raise RuntimeError("kernel failed")

    with pytest.raises(RuntimeError, match="kernel failed"):
        run_account_sessions(
            _contexts("acc1"),
            broken,
            requested_budget_seconds=21_600,
            max_sessions=None,
            check_only=False,
            log=log.append,
            read_quota=QuotaBook(acc1=30.0),
        )

    assert any("session=1 error=RuntimeError: kernel failed" in line for line in log)


@pytest.mark.parametrize(
    ("budget", "max_sessions", "message"),
    [(3_599, None, "at least 3600"), (21_600, 0, "--max-runs")],
)
def test_session_arguments_are_validated(tmp_path, budget, max_sessions, message):
    book = QuotaBook(acc1=30.0)

    with pytest.raises(ValueError, match=message):
        run_account_sessions(
            _contexts("acc1"),
            ScriptedSessions(tmp_path, book, [1]),
            requested_budget_seconds=budget,
            max_sessions=max_sessions,
            check_only=False,
            log=_quiet,
            read_quota=book,
        )
```

Trace for the quota-exhausted test: session 1 runs on acc1 (7.0 h) and spends 6 h; then acc1 has 1.0 h and acc2 1.2 h, and the best budget is `1.2 h − 0.5 h = 2,520 s < 3,600 s`.

- [ ] **Step 2: Write the failing service tests**

In `seed-pipeline/tests/integrations/kaggle/test_service.py` add to the imports `import json`, `from seed_pipeline.integrations.kaggle.models import KernelPresence, KernelRemoteState, KernelStatus` (merge into the existing models import) and `resolve_session_contexts` to the `service` import, then append:

```python
def test_session_contexts_for_auto_are_every_profile(tmp_path, monkeypatch):
    env_file = _write_profiles(tmp_path)
    _clear_kaggle_environment(monkeypatch)

    contexts = resolve_session_contexts("auto", env_file=env_file)

    assert [c.profile.name for c in contexts if c.profile] == ["acc1", "acc2"]


@pytest.mark.parametrize(("account", "expected"), [("acc2", "acc2"), (None, "acc1")])
def test_session_contexts_for_one_account(tmp_path, monkeypatch, account, expected):
    env_file = _write_profiles(tmp_path)
    _clear_kaggle_environment(monkeypatch)

    contexts = resolve_session_contexts(account, env_file=env_file)

    assert [c.profile.name for c in contexts if c.profile] == [expected]


def test_session_contexts_require_account_profiles(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "KAGGLE_USERNAME=legacy-user\nKAGGLE_API_TOKEN=legacy-token\n",
        encoding="utf-8",
    )
    _clear_kaggle_environment(monkeypatch)

    with pytest.raises(ValueError, match="need account profiles"):
        resolve_session_contexts(None, env_file=env_file)


def test_active_kernel_profile_finds_the_account_running_the_job(tmp_path, monkeypatch):
    env_file = _write_profiles(tmp_path)
    _clear_kaggle_environment(monkeypatch)
    contexts = resolve_profile_execution_contexts(env_file=env_file)
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "query": "query",
                "candidates": [{"chunk_id": "c1", "document_text": "text"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    inspected: list[str] = []

    class FakeKernelService:
        def __init__(self, runner, owner):
            self.owner = owner

        def inspect_state(self, reference):
            inspected.append(reference)
            status = (
                KernelStatus.RUNNING
                if self.owner == "secondary-user"
                else KernelStatus.COMPLETE
            )
            return KernelRemoteState(reference, KernelPresence.EXISTS, status)

    monkeypatch.setattr(kaggle_service, "KernelService", FakeKernelService)

    profile = kaggle_service.active_kernel_profile(
        contexts,
        stage=StageName.RERANK,
        model=MODEL,
        input_path=candidates,
        output_dir=tmp_path / "output",
        runtime_profile=rerank_runtime_profile(MODEL),
    )

    assert profile == "acc2"
    assert [reference.split("/")[0] for reference in inspected] == [
        "primary-user",
        "secondary-user",
    ]
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest -q tests/integrations/kaggle/test_sessions.py tests/integrations/kaggle/test_service.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'seed_pipeline.integrations.kaggle.sessions'` and `ImportError: cannot import name 'resolve_session_contexts'`.

- [ ] **Step 4: Add session contexts and active kernel discovery to the service**

In `seed-pipeline/src/seed_pipeline/integrations/kaggle/service.py` add `from collections.abc import Sequence` and extend the models import to `KernelPresence, KernelStatus, PipelineResult, StageName, StageRequest`. Below `DEFAULT_ENV_PATH = PROJECT_ENV_FILE` add `AUTO_ACCOUNT = "auto"`, and below `resolve_profile_execution_contexts` add:

```python
def resolve_session_contexts(
    kaggle_account: str | None,
    *,
    env_file: Path = DEFAULT_ENV_PATH,
) -> tuple[KaggleExecutionContext, ...]:
    """Accounts a quota-aware session may use: every profile for `auto`, else one."""
    if kaggle_account is not None and kaggle_account.strip().casefold() == AUTO_ACCOUNT:
        contexts = resolve_profile_execution_contexts(env_file=env_file)
    else:
        contexts = (resolve_execution_context(kaggle_account, env_file=env_file),)
    if not contexts or any(context.profile is None for context in contexts):
        raise ValueError(
            "Kaggle sessions need account profiles: set KAGGLE_ACCOUNT_DEFAULT, "
            "KAGGLE_SHARED_OWNER and KAGGLE_ACC<N>_USERNAME/KAGGLE_ACC<N>_API_TOKEN "
            "in seed-pipeline/.env"
        )
    return contexts


def active_kernel_profile(
    contexts: Sequence[KaggleExecutionContext],
    *,
    stage: StageName,
    model: str,
    input_path: Path,
    output_dir: Path,
    gguf_root: Path = GGUF_ROOT,
    runtime_profile: RuntimeCandidate | None = None,
) -> str | None:
    """Profile whose account already queues or runs this job's kernel, if any.

    A command restarted after a crash or reboot attaches to that kernel instead of
    starting a duplicate on the account with the most quota.
    """
    profiled = [context for context in contexts if context.profile is not None]
    if not profiled:
        return None
    job = get_stage_adapter(stage).build_job(
        StageRequest(
            stage,
            model,
            Path(input_path),
            Path(output_dir),
            Path(gguf_root),
            profiled[0].owners,
            runtime_profile=runtime_profile,
        )
    )
    for context in profiled:
        assert context.profile is not None
        kernels = PipelineKernelService(
            KernelService(context.runner, context.owners.execution),
            owner=context.owners.execution,
            source_root=PROJECT_ROOT / "src",
        )
        state = kernels.discover(job)
        if state.presence is KernelPresence.EXISTS and state.status in {
            KernelStatus.QUEUED,
            KernelStatus.RUNNING,
        }:
            return context.profile.name
    return None
```

- [ ] **Step 5: Implement the session runner**

Create `seed-pipeline/src/seed_pipeline/integrations/kaggle/sessions.py`:

```python
"""Run a Kaggle stage as consecutive GPU sessions on the account with the most quota."""

from __future__ import annotations

from collections.abc import Callable, Generator, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass
from enum import StrEnum

from seed_pipeline.integrations.kaggle.job_lock import (
    KaggleAccountBusy,
    kaggle_account_lock,
)
from seed_pipeline.integrations.kaggle.models import PipelineResult
from seed_pipeline.integrations.kaggle.quota import (
    MINIMUM_SESSION_SECONDS,
    AccountQuota,
    SessionAccount,
    quota_table,
    read_account_quota,
    select_session_account,
)
from seed_pipeline.integrations.kaggle.service import KaggleExecutionContext

QuotaReader = Callable[[KaggleExecutionContext], AccountQuota]
AccountLock = Callable[[str], AbstractContextManager[str]]
SessionLog = Callable[[str], None]


class QuotaExhausted(RuntimeError):
    """No free account has at least one hour of GPU quota for a session."""

    def __init__(self, quotas: tuple[AccountQuota, ...], locked: frozenset[str]):
        super().__init__(
            "no free Kaggle account has at least one hour of GPU quota for a session"
        )
        self.quotas = quotas
        self.locked = locked

    @property
    def table(self) -> tuple[str, ...]:
        return quota_table(self.quotas, locked=self.locked)


@dataclass(frozen=True)
class ReservedAccount:
    context: KaggleExecutionContext
    quota: AccountQuota
    budget_seconds: int

    @property
    def profile(self) -> str:
        return self.quota.profile


class StopReason(StrEnum):
    COMPLETE = "complete"
    QUOTA_EXHAUSTED = "quota-exhausted"
    MAX_RUNS = "max-runs"
    NO_PROGRESS = "no-progress"
    CHECK_ONLY = "check-only"


@dataclass(frozen=True)
class SessionRunResult:
    result: PipelineResult | None
    sessions: int
    stop_reason: StopReason
    quota_table: tuple[str, ...] = ()


def _profile_name(context: KaggleExecutionContext) -> str:
    if context.profile is None:
        raise ValueError("Kaggle sessions need account profiles (acc1, acc2, ...)")
    return context.profile.name


def _format_quota(quota: AccountQuota) -> str:
    refresh = quota.refresh_at.isoformat() if quota.refresh_at else "unknown"
    return (
        f"quota account={quota.profile} remaining={quota.remaining_hours:.2f}h "
        f"total={quota.total_hours:.2f}h refresh_at={refresh}"
    )


def _lock_preferred(
    profile: str | None,
    *,
    requested_budget_seconds: int,
    stack: ExitStack,
    locker: AccountLock,
    log: SessionLog,
) -> SessionAccount | None:
    if profile is None:
        return None
    try:
        stack.enter_context(locker(profile))
    except KaggleAccountBusy:
        log(
            f"account={profile} runs this job's kernel but is locked by another "
            "local job; choosing by quota"
        )
        return None
    log(f"account={profile} already runs this job's kernel; attaching")
    return SessionAccount(profile, requested_budget_seconds)


def _lock_best(
    quotas: tuple[AccountQuota, ...],
    *,
    requested_budget_seconds: int,
    stack: ExitStack,
    locker: AccountLock,
    log: SessionLog,
) -> SessionAccount:
    busy: set[str] = set()
    while True:
        choice = select_session_account(
            quotas,
            locked=frozenset(busy),
            requested_budget_seconds=requested_budget_seconds,
        )
        if choice is None:
            raise QuotaExhausted(quotas, frozenset(busy))
        try:
            stack.enter_context(locker(choice.profile))
        except KaggleAccountBusy:
            busy.add(choice.profile)
            log(f"account={choice.profile} is locked by another local job; skipped")
            continue
        return choice


@contextmanager
def reserve_session_account(
    contexts: Sequence[KaggleExecutionContext],
    *,
    requested_budget_seconds: int,
    log: SessionLog,
    read_quota: QuotaReader | None = None,
    account_lock: AccountLock | None = None,
    preferred_profile: str | None = None,
) -> Generator[ReservedAccount, None, None]:
    """Read every account's GPU quota, then lock one account for one session."""
    reader = read_quota or read_account_quota
    locker = account_lock or kaggle_account_lock
    by_profile = {_profile_name(context): context for context in contexts}
    quotas = tuple(reader(context) for context in contexts)
    for quota in quotas:
        log(_format_quota(quota))
    with ExitStack() as stack:
        choice = _lock_preferred(
            preferred_profile,
            requested_budget_seconds=requested_budget_seconds,
            stack=stack,
            locker=locker,
            log=log,
        ) or _lock_best(
            quotas,
            requested_budget_seconds=requested_budget_seconds,
            stack=stack,
            locker=locker,
            log=log,
        )
        quota = next(item for item in quotas if item.profile == choice.profile)
        log(
            f"account={choice.profile} username={quota.username} "
            f"budget_seconds={choice.budget_seconds}"
        )
        yield ReservedAccount(by_profile[choice.profile], quota, choice.budget_seconds)


def _run_one_session(
    contexts: Sequence[KaggleExecutionContext],
    run_session: Callable[[ReservedAccount, int], PipelineResult],
    index: int,
    *,
    requested_budget_seconds: int,
    check_only: bool,
    log: SessionLog,
    reader: QuotaReader,
    account_lock: AccountLock | None,
    preferred_profile: str | None,
) -> PipelineResult:
    with reserve_session_account(
        contexts,
        requested_budget_seconds=requested_budget_seconds,
        log=log,
        read_quota=reader,
        account_lock=account_lock,
        preferred_profile=preferred_profile,
    ) as reserved:
        log(
            f"session={index} start account={reserved.profile} "
            f"budget_seconds={reserved.budget_seconds}"
        )
        try:
            result = run_session(reserved, index)
        except BaseException as error:
            log(f"session={index} error={type(error).__name__}: {error}")
            raise
        for action in result.actions:
            log(
                f"session={index} {action.verb.value} {action.resource_kind} "
                f"{action.reference}: {action.reason}"
            )
        log(
            f"session={index} end "
            f"pairs={result.completion.complete}/{result.completion.total}"
        )
        if not check_only:
            after = reader(reserved.context)
            log(
                f"session={index} quota_after account={after.profile} "
                f"remaining={after.remaining_hours:.2f}h"
            )
        return result


def run_account_sessions(
    contexts: Sequence[KaggleExecutionContext],
    run_session: Callable[[ReservedAccount, int], PipelineResult],
    *,
    requested_budget_seconds: int,
    max_sessions: int | None,
    check_only: bool,
    log: SessionLog,
    read_quota: QuotaReader | None = None,
    account_lock: AccountLock | None = None,
    preferred_profile: str | None = None,
) -> SessionRunResult:
    """Run sessions until complete, out of quota, without progress or at max_sessions."""
    if requested_budget_seconds < MINIMUM_SESSION_SECONDS:
        raise ValueError(
            f"--budget-seconds must be at least {MINIMUM_SESSION_SECONDS} "
            "for Kaggle sessions"
        )
    if max_sessions is not None and max_sessions < 1:
        raise ValueError("--max-runs must be at least 1")
    if not contexts:
        raise ValueError("at least one Kaggle account profile is required")
    reader = read_quota or read_account_quota
    sessions = 0
    last: PipelineResult | None = None
    while max_sessions is None or sessions < max_sessions:
        index = sessions + 1
        try:
            result = _run_one_session(
                contexts,
                run_session,
                index,
                requested_budget_seconds=requested_budget_seconds,
                check_only=check_only,
                log=log,
                reader=reader,
                account_lock=account_lock,
                preferred_profile=preferred_profile if index == 1 else None,
            )
        except QuotaExhausted as exhausted:
            log(f"stop=quota-exhausted sessions={sessions}; GPU quota per account:")
            for row in exhausted.table:
                log(row)
            return SessionRunResult(
                last, sessions, StopReason.QUOTA_EXHAUSTED, exhausted.table
            )
        sessions = index
        previous = last
        last = result
        if check_only:
            return SessionRunResult(result, sessions, StopReason.CHECK_ONLY)
        if result.completion.is_complete:
            log(f"stop=complete sessions={sessions}")
            return SessionRunResult(result, sessions, StopReason.COMPLETE)
        if (
            previous is not None
            and result.completion.complete <= previous.completion.complete
        ):
            log(
                f"stop=no-progress sessions={sessions} "
                f"pairs={result.completion.complete}/{result.completion.total}"
            )
            return SessionRunResult(result, sessions, StopReason.NO_PROGRESS)
    log(f"stop=max-runs sessions={sessions}")
    return SessionRunResult(last, sessions, StopReason.MAX_RUNS)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest -q tests/integrations/kaggle/test_sessions.py tests/integrations/kaggle/test_service.py`
Expected: PASS.

- [ ] **Step 7: Run the seed gate**

Run the **Seed gate**. Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add seed-pipeline/src/seed_pipeline/integrations/kaggle/sessions.py \
  seed-pipeline/src/seed_pipeline/integrations/kaggle/service.py \
  seed-pipeline/tests/integrations/kaggle/test_sessions.py \
  seed-pipeline/tests/integrations/kaggle/test_service.py
git diff --cached --name-status
git commit -m "feat(seed): run Kaggle sessions on the account with the most GPU quota

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: `seed rerank --backend kaggle --kaggle-account auto`

**Files:**
- Modify: `seed-pipeline/src/seed_pipeline/evaluation/rerank_service.py`
- Modify: `seed-pipeline/src/seed_pipeline/cli/commands/rerank.py`
- Modify: `seed-pipeline/docs/guides/workflow-local-kaggle.md`, `seed-pipeline/docs/guides/workflow-local-only.md`, `seed-pipeline/docs/guides/cli-reference.md`
- Test: `seed-pipeline/tests/evaluation/test_rerank_service.py`, `seed-pipeline/tests/cli/test_rerank_command.py`

**Interfaces:**
- Consumes: `resolve_session_contexts`, `active_kernel_profile`, `runtime_manifest_sha256(runner, owners)`, `run_kaggle_stage(..., resume_remote, artifact_sink, local_checkpoint)` (Tasks 7–8); `reserve_session_account`, `run_account_sessions`, `QuotaExhausted`, `ReservedAccount` (Task 8); `RerankCacheCheckpoint` (Task 5); `open_rerank_log` (Task 1); from Plan A: `ensure_runtime_profile(*, workload, benchmark_stage, model, input_path, gguf_root, budget_seconds, dry_run, force, kaggle_account=None, runtime_sha256=None, benchmark_runner=None, ...)` calling `benchmark_runner(**keyword_arguments)` whose keywords are `run_kaggle_stage` parameters.
- Produces:
  - `RerankRequest.max_runs: int | None = None`; `RerankStageResult.quota: tuple[str, ...] = ()`.
  - `KaggleRerankBackend._run_kaggle_unlocked(request: RerankRequest) -> RerankStageResult` (signature unchanged): finalizes from a complete local cache without Kaggle; resolves session contexts (`auto` = every profile); runs the runtime benchmark, if one is needed, on a reserved account; attaches first to an account that already runs this job's kernel; runs sessions with `max_runs=1` each, `force` on session 1 only and `resume_remote=not request.force`; merges every artifact into the local cache through the sink; returns `incomplete=True` with `actions` containing `missing_pairs=N`, `target=...`, `sessions=N`, `stop=<reason>` and `quota` set when quota ran out; logs to `data/work/logs/rerank/<model-slug>.log`.
  - CLI: `--kaggle-account accN|auto`, `--max-runs N` (`min=1`, default unlimited); every `seed rerank` appends `command ...` start, status and error lines to the model log; the JSON/text result includes `quota`.

- [ ] **Step 1: Write the failing backend tests**

In `seed-pipeline/tests/evaluation/test_rerank_service.py`:

Add these imports (then run `uv run ruff check --fix tests/evaluation/test_rerank_service.py` to sort them):

```python
from datetime import datetime

from tests.integrations.kaggle.factories import job_identity, stage_job

from seed_pipeline.config.paths import rerank_log_path
from seed_pipeline.evaluation.rerank_cache_checkpoint import RerankCacheCheckpoint
from seed_pipeline.integrations.kaggle import sessions
from seed_pipeline.integrations.kaggle.api import KaggleCommandRunner
from seed_pipeline.integrations.kaggle.config import (
    KaggleAccountProfile,
    OwnerConfiguration,
)
from seed_pipeline.integrations.kaggle.models import CloudArtifact
from seed_pipeline.integrations.kaggle.quota import AccountQuota
from seed_pipeline.integrations.kaggle.service import KaggleExecutionContext
```

Add below `fake_local_backend`:

```python
MODEL = "qwen3-reranker:0.6b-fp16"


def _session_contexts(*names: str) -> tuple[KaggleExecutionContext, ...]:
    return tuple(
        KaggleExecutionContext(
            KaggleAccountProfile(name, f"user-{name}", f"token-{name}"),
            OwnerConfiguration(
                f"user-{name}", "user-acc1", "user-acc1", f"user-{name}"
            ),
            KaggleCommandRunner(environment={}),
        )
        for name in names
    )


def _scores_artifact(cache: RerankScoreCache) -> CloudArtifact:
    return CloudArtifact(
        cache.path,
        cache.path.with_name("manifest.json"),
        job_identity(),
        Completion(1, 1, 0),
    )


@pytest.fixture
def kaggle_sessions(monkeypatch, tmp_path) -> dict[str, float]:
    """Fake Kaggle: GPU hours left per profile and a reusable runtime profile."""
    spec = require_model(MODEL)
    assert spec.rerank_search_space is not None
    selected = spec.rerank_search_space.candidates[0]
    quotas = {"acc1": 27.58, "acc2": 29.61}

    def read_quota(context: KaggleExecutionContext) -> AccountQuota:
        assert context.profile is not None
        return AccountQuota(
            context.profile.name,
            context.profile.username,
            quotas[context.profile.name],
            30.0,
            datetime(2026, 9, 19),
        )

    monkeypatch.setattr(rerank_service, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(
        kaggle_service,
        "resolve_session_contexts",
        lambda _account: _session_contexts(*quotas),
    )
    monkeypatch.setattr(
        kaggle_service, "runtime_manifest_sha256", lambda *_args, **_kwargs: "c" * 64
    )
    monkeypatch.setattr(
        kaggle_service, "active_kernel_profile", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        auto_profile,
        "ensure_runtime_profile",
        lambda **_kwargs: SimpleNamespace(
            profile=SimpleNamespace(selected=selected), action="reuse"
        ),
    )
    monkeypatch.setattr(sessions, "read_account_quota", read_quota)
    return quotas
```

In `test_kaggle_dry_run_formats_resource_identity_and_does_not_register`, add as the second line of the body:

```python
    monkeypatch.setattr(
        kaggle_service,
        "resolve_session_contexts",
        lambda _account: _session_contexts("acc1"),
    )
```

Delete `test_kaggle_rerank_propagates_selected_account` and append:

```python
def test_kaggle_rerank_merges_session_scores_and_registers_the_variant(
    complete_run, complete_rerank_cache, kaggle_sessions, monkeypatch, tmp_path
):
    calls: list[dict] = []

    def fake_stage(**kwargs):
        calls.append(kwargs)
        kwargs["artifact_sink"](
            stage_job(tmp_path / "job"), _scores_artifact(complete_rerank_cache)
        )
        return SimpleNamespace(
            completion=Completion(1, 1, 0), actions=(), artifact_path=None
        )

    monkeypatch.setattr(kaggle_service, "run_kaggle_stage", fake_stage)

    result = KaggleRerankBackend().run(
        replace(
            request(complete_run, MODEL), kaggle_account="auto", budget_seconds=21_600
        )
    )

    assert not result.incomplete
    assert result.artifact_dir == complete_run / "rerank" / "qwen3_reranker_0_6b_fp16"
    assert [
        (
            call["kaggle_account"],
            call["budget_seconds"],
            call["max_runs"],
            call["force"],
            call["resume_remote"],
        )
        for call in calls
    ] == [("acc2", 21_600, 1, False, True)]
    assert isinstance(calls[0]["local_checkpoint"], RerankCacheCheckpoint)
    assert "stop=complete" in result.actions
    log = rerank_log_path(MODEL).read_text(encoding="utf-8")
    assert "quota account=acc1 remaining=27.58h" in log
    assert "session=1 start account=acc2" in log
    assert "merged artifact pairs=1/1" in log


def test_kaggle_rerank_stops_with_a_quota_table_when_no_account_has_an_hour(
    complete_run, kaggle_sessions, monkeypatch
):
    kaggle_sessions.update(acc1=1.2, acc2=0.4)

    def no_session(**_kwargs):
        raise AssertionError("no Kaggle session may start without quota")

    monkeypatch.setattr(kaggle_service, "run_kaggle_stage", no_session)

    result = KaggleRerankBackend().run(
        replace(
            request(complete_run, MODEL), kaggle_account="auto", budget_seconds=21_600
        )
    )

    assert result.incomplete
    assert "stop=quota-exhausted" in result.actions
    assert "missing_pairs=1" in result.actions
    assert len(result.quota) == 3
    assert "2026-09-19T00:00:00" in result.quota[1]
    assert load_run_record(complete_run / "run.json").rerank_variants == {}


def test_kaggle_rerank_attaches_to_the_account_already_running_the_kernel(
    complete_run, complete_rerank_cache, kaggle_sessions, monkeypatch, tmp_path
):
    accounts: list[str] = []

    def fake_stage(**kwargs):
        accounts.append(kwargs["kaggle_account"])
        kwargs["artifact_sink"](
            stage_job(tmp_path / "job"), _scores_artifact(complete_rerank_cache)
        )
        return SimpleNamespace(
            completion=Completion(1, 1, 0), actions=(), artifact_path=None
        )

    monkeypatch.setattr(
        kaggle_service, "active_kernel_profile", lambda *_args, **_kwargs: "acc1"
    )
    monkeypatch.setattr(kaggle_service, "run_kaggle_stage", fake_stage)

    KaggleRerankBackend().run(
        replace(
            request(complete_run, MODEL), kaggle_account="auto", budget_seconds=21_600
        )
    )

    assert accounts == ["acc1"]


def test_forced_kaggle_rerank_forces_only_the_first_session(
    complete_run, complete_rerank_cache, kaggle_sessions, monkeypatch, tmp_path
):
    calls: list[tuple[bool, bool]] = []

    def fake_stage(**kwargs):
        calls.append((kwargs["force"], kwargs["resume_remote"]))
        if len(calls) == 1:
            return SimpleNamespace(
                completion=Completion(1, 0, 1), actions=(), artifact_path=None
            )
        kwargs["artifact_sink"](
            stage_job(tmp_path / "job"), _scores_artifact(complete_rerank_cache)
        )
        return SimpleNamespace(
            completion=Completion(1, 1, 0), actions=(), artifact_path=None
        )

    monkeypatch.setattr(kaggle_service, "run_kaggle_stage", fake_stage)

    result = KaggleRerankBackend().run(
        replace(
            request(complete_run, MODEL, force=True),
            kaggle_account="auto",
            budget_seconds=21_600,
        )
    )

    assert calls == [(True, False), (False, False)]
    assert not result.incomplete


def test_kaggle_rerank_finalizes_a_complete_local_cache_without_kaggle(
    complete_run, complete_rerank_cache, monkeypatch, tmp_path
):
    def no_accounts(_account):
        raise AssertionError("a complete local cache needs no Kaggle account")

    monkeypatch.setattr(rerank_service, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(
        rerank_service,
        "rerank_score_cache_path",
        lambda _model: complete_rerank_cache.path,
    )
    monkeypatch.setattr(kaggle_service, "resolve_session_contexts", no_accounts)

    result = KaggleRerankBackend().run(request(complete_run, MODEL))

    assert result.artifact_dir == complete_run / "rerank" / "qwen3_reranker_0_6b_fp16"
    assert "reuse=local score cache" in result.actions
```

- [ ] **Step 2: Write the failing CLI tests**

In `seed-pipeline/tests/cli/test_rerank_command.py`: change the import `from seed_pipeline.config.paths import run_dir` to `from seed_pipeline.config.paths import rerank_log_path, run_dir`, add `quota=(),` after `benchmark_levels=0,` in `fake_backend`, and append:

```python
def test_rerank_passes_auto_account_and_max_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        rerank_command, "KaggleRerankBackend", fake_backend(captured, incomplete=True)
    )

    result = runner.invoke(
        app,
        [
            "rerank",
            "--run",
            "experiment",
            "--backend",
            "kaggle",
            "--kaggle-account",
            "auto",
            "--max-runs",
            "4",
        ],
    )

    assert result.exit_code == 3
    assert captured["request"].kaggle_account == "auto"
    assert captured["request"].max_runs == 4


def test_rerank_rejects_zero_max_runs() -> None:
    result = runner.invoke(
        app, ["rerank", "--run", "experiment", "--backend", "kaggle", "--max-runs", "0"]
    )

    assert result.exit_code == 2


def test_rerank_appends_command_lines_to_the_model_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        rerank_command, "LocalRerankBackend", fake_backend(captured, incomplete=False)
    )

    result = runner.invoke(
        app, ["rerank", "--run", "experiment", "--model", "qwen3-reranker:0.6b-fp16"]
    )

    assert result.exit_code == 0, result.output
    log = rerank_log_path("qwen3-reranker:0.6b-fp16").read_text(encoding="utf-8")
    assert "command backend=local run=experiment" in log
    assert "command status=complete actions=missing_pairs=0" in log


def test_rerank_logs_a_failed_command(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingBackend:
        def run(self, request):
            raise RuntimeError("boom")

    monkeypatch.setattr(rerank_command, "LocalRerankBackend", FailingBackend)

    result = runner.invoke(
        app, ["rerank", "--run", "experiment", "--model", "qwen3-reranker:0.6b-fp16"]
    )

    assert result.exit_code == 1
    log = rerank_log_path("qwen3-reranker:0.6b-fp16").read_text(encoding="utf-8")
    assert "command error=RuntimeError: boom" in log
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest -q tests/evaluation/test_rerank_service.py tests/cli/test_rerank_command.py`
Expected: FAIL — `RerankRequest` has no `max_runs`, `RerankStageResult` has no `quota`, `No such option: --max-runs`, and the Kaggle backend still calls `run_kaggle_stage` without sessions.

- [ ] **Step 4: Run the Kaggle backend as quota-aware sessions**

In `seed-pipeline/src/seed_pipeline/evaluation/rerank_service.py`:

Change the imports to add:

```python
from typing import Any

from seed_pipeline.artifacts.bundle import ArtifactBundle, load_bundle
from seed_pipeline.cache.jsonl_records import ValidatedSubset, merge_records
from seed_pipeline.evaluation.rerank_cache_checkpoint import RerankCacheCheckpoint
from seed_pipeline.evaluation.rerank_log import open_rerank_log
```

(replacing the existing `load_bundle` and `merge_records` imports).

Add `max_runs: int | None = None` as the last field of `RerankRequest` and `quota: tuple[str, ...] = ()` as the last field of `RerankStageResult`.

Add below `_existing_variant`:

```python
def _local_subset(
    cache_path: Path,
    candidates: Path,
    model: str,
    *,
    model_sha256: str,
    request_contract_sha256: str,
) -> ValidatedSubset:
    with kaggle_cache_lock(cache_path):
        return RerankScoreCache(
            cache_path,
            model_sha256=model_sha256,
            request_contract_sha256=request_contract_sha256,
        ).validate_subset(candidates, model)


def _finalize_variant(
    request: RerankRequest,
    workspace: RunWorkspace,
    candidate_bundle: ArtifactBundle,
    cache_path: Path,
    identity: RerankVariantIdentity,
    subset: ValidatedSubset,
    actions: tuple[str, ...],
) -> RerankStageResult:
    # Different model jobs may finish concurrently; serialize the run registry update.
    with kaggle_cache_lock(request.run_root / "run.json"):
        bundle = finalize_run_rerank_bundle(
            workspace=workspace,
            candidate_bundle=candidate_bundle,
            cache_path=cache_path,
            identity=identity,
            force=request.force,
        )
    return RerankStageResult(bundle.root, identity.sha256, subset.sha256, actions)
```

Replace `KaggleRerankBackend._run_kaggle_unlocked` with:

```python
    @staticmethod
    def _run_kaggle_unlocked(request: RerankRequest) -> RerankStageResult:
        from seed_pipeline.integrations.kaggle.auto_profile import (
            ensure_runtime_profile,
        )
        from seed_pipeline.integrations.kaggle.models import (
            CloudArtifact,
            PipelineResult,
            StageJob,
            StageName,
        )
        from seed_pipeline.integrations.kaggle.service import (
            active_kernel_profile,
            resolve_session_contexts,
            run_kaggle_stage,
            runtime_manifest_sha256,
        )
        from seed_pipeline.integrations.kaggle.sessions import (
            QuotaExhausted,
            ReservedAccount,
            reserve_session_account,
            run_account_sessions,
        )

        spec = require_model(request.model)
        if spec.kind is not ModelKind.RERANKER:
            raise ValueError("--model must select a reranker model")
        workspace = _workspace(request)
        candidate_bundle = load_bundle(
            workspace.candidates_dir,
            expected_type="retrieval_candidates",
            require_complete=True,
        )
        identity = RerankVariantIdentity.create(
            candidate_bundle.manifest.data_sha256, request.model
        )
        existing = _existing_variant(
            workspace, spec.slug, identity.sha256, force=request.force
        )
        if existing is not None:
            bundle = load_registered_rerank_bundle(workspace, existing)
            return RerankStageResult(
                bundle.root,
                identity.sha256,
                bundle.manifest.data_sha256,
                ("reuse=complete variant",),
            )
        if spec.rerank_contract is None:
            raise ValueError(f"Reranker {request.model} has no scoring contract")
        contract = spec.rerank_contract.sha256
        cache_path = rerank_score_cache_path(request.model)
        candidates = candidate_bundle.data_path
        log = open_rerank_log(request.model, echo=True)

        def local_subset() -> ValidatedSubset:
            return _local_subset(
                cache_path,
                candidates,
                request.model,
                model_sha256=spec.sha256,
                request_contract_sha256=contract,
            )

        if request.force and not request.dry_run:
            with kaggle_cache_lock(cache_path):
                cache = RerankScoreCache(
                    cache_path,
                    model_sha256=spec.sha256,
                    request_contract_sha256=contract,
                )
                cache.replace_keys(
                    cache.expected_keys_from_candidates(candidates, request.model)
                )
            log("force: removed this run's pairs from the local score cache")
        subset = local_subset()
        missing_pairs = f"missing_pairs={subset.missing}"
        target = f"target={workspace.rerank_dir(spec.slug)}"
        log(
            f"kaggle start run={request.run_root.name} "
            f"account={request.kaggle_account or 'default'} {missing_pairs}"
        )
        if subset.is_complete and not request.dry_run:
            log("local score cache is complete; no Kaggle session needed")
            return _finalize_variant(
                request,
                workspace,
                candidate_bundle,
                cache_path,
                identity,
                subset,
                (missing_pairs, "reuse=local score cache"),
            )

        contexts = resolve_session_contexts(request.kaggle_account)
        first = contexts[0]

        def benchmark_runner(**benchmark_arguments: Any) -> PipelineResult:
            with reserve_session_account(
                contexts, requested_budget_seconds=request.budget_seconds, log=log
            ) as reserved:
                return run_kaggle_stage(
                    stage=StageName.RERANK_BENCHMARK,
                    model=request.model,
                    output_dir=WORK_DIR / "kaggle-runtime-benchmarks" / spec.slug,
                    budget_seconds=reserved.budget_seconds,
                    kaggle_account=reserved.profile,
                    **benchmark_arguments,
                )

        try:
            resolution = ensure_runtime_profile(
                workload="rerank",
                benchmark_stage=StageName.RERANK_BENCHMARK.value,
                model=request.model,
                input_path=candidates,
                gguf_root=GGUF_ROOT,
                budget_seconds=request.budget_seconds,
                dry_run=request.dry_run,
                force=request.force,
                kaggle_account=first.profile.name if first.profile else None,
                runtime_sha256=(
                    None
                    if request.dry_run
                    else runtime_manifest_sha256(first.runner, first.owners)
                ),
                benchmark_runner=benchmark_runner,
            )
        except QuotaExhausted as exhausted:
            for row in exhausted.table:
                log(row)
            return RerankStageResult(
                None,
                identity.sha256,
                None,
                (missing_pairs, target, "stop=quota-exhausted"),
                incomplete=True,
                quota=exhausted.table,
            )
        if resolution.profile is None:
            return RerankStageResult(
                None,
                identity.sha256,
                None,
                (missing_pairs, f"profile={resolution.action}"),
                incomplete=True,
            )
        runtime_profile = resolution.profile.selected
        remote_dir = WORK_DIR / "kaggle-rerank-scores" / spec.slug

        def merge_artifact(_job: StageJob, artifact: CloudArtifact) -> None:
            remote = RerankScoreCache(
                artifact.data_path,
                model_sha256=spec.sha256,
                request_contract_sha256=contract,
            )
            with kaggle_cache_lock(cache_path):
                _merge_remote_rerank_scores(
                    cache_path,
                    remote,
                    model_sha256=spec.sha256,
                    request_contract_sha256=contract,
                )
            merged = local_subset()
            log(
                f"merged artifact pairs={artifact.completion.complete}/"
                f"{artifact.completion.total}; local cache pairs="
                f"{merged.complete}/{merged.total}"
            )

        checkpoint_source = RerankCacheCheckpoint(cache_path)

        def run_session(reserved: ReservedAccount, index: int) -> PipelineResult:
            return run_kaggle_stage(
                stage=StageName.RERANK,
                model=request.model,
                input_path=candidates,
                output_dir=remote_dir,
                gguf_root=GGUF_ROOT,
                force=request.force and index == 1,
                resume_remote=not request.force,
                check_only=request.dry_run,
                max_runs=1,
                budget_seconds=reserved.budget_seconds,
                runtime_profile=runtime_profile,
                kaggle_account=reserved.profile,
                artifact_sink=merge_artifact,
                local_checkpoint=checkpoint_source,
            )

        preferred = (
            None
            if request.dry_run or request.force
            else active_kernel_profile(
                contexts,
                stage=StageName.RERANK,
                model=request.model,
                input_path=candidates,
                output_dir=remote_dir,
                gguf_root=GGUF_ROOT,
                runtime_profile=runtime_profile,
            )
        )
        outcome = run_account_sessions(
            contexts,
            run_session,
            requested_budget_seconds=request.budget_seconds,
            max_sessions=request.max_runs,
            check_only=request.dry_run,
            log=log,
            preferred_profile=preferred,
        )
        subset = local_subset()
        actions = (
            f"missing_pairs={subset.missing}",
            target,
            f"sessions={outcome.sessions}",
            f"stop={outcome.stop_reason.value}",
            *(
                f"{action.verb.value} {action.resource_kind} "
                f"{action.reference}: {action.reason}"
                for action in (outcome.result.actions if outcome.result else ())
            ),
        )
        if request.dry_run or not subset.is_complete:
            return RerankStageResult(
                None,
                identity.sha256,
                None,
                actions,
                incomplete=True,
                quota=outcome.quota_table,
            )
        result = _finalize_variant(
            request, workspace, candidate_bundle, cache_path, identity, subset, actions
        )
        if outcome.result is not None and outcome.result.artifact_path is not None:
            _cleanup_completed_stage_artifact(outcome.result.artifact_path, remote_dir)
        log(f"kaggle complete variant={result.artifact_dir}")
        return result
```

If Plan A changed how `ensure_runtime_profile` is called in this method (new arguments, a different benchmark stage), keep Plan A's arguments and add only `runtime_sha256` and `benchmark_runner` as shown.

- [ ] **Step 5: Add the CLI options and command log**

Replace `seed-pipeline/src/seed_pipeline/cli/commands/rerank.py` with the version below. If Plan A added options (for example `--benchmark`) or another adapter, keep them and route the chosen adapter's `run` through `_logged` the same way:

```python
from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, cast

import typer

from seed_pipeline.cli.options import Backend
from seed_pipeline.cli.runtime import (
    CommandResult,
    CommandStatus,
    run_handler,
    state_from_context,
)
from seed_pipeline.config.defaults import (
    DEFAULT_BUDGET_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_RERANKER_MODEL,
)
from seed_pipeline.config.paths import run_dir
from seed_pipeline.evaluation.rerank_log import open_rerank_log
from seed_pipeline.evaluation.rerank_service import (
    KaggleRerankBackend,
    LocalRerankBackend,
    RerankRequest,
    RerankStageResult,
)


def rerank(
    ctx: typer.Context,
    run: Annotated[str, typer.Option("--run")] = cast(str, ...),
    backend: Annotated[Backend, typer.Option("--backend")] = Backend.LOCAL,
    model: Annotated[str, typer.Option("--model")] = DEFAULT_RERANKER_MODEL,
    force: Annotated[bool, typer.Option("--force")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    budget_seconds: Annotated[
        int, typer.Option("--budget-seconds")
    ] = DEFAULT_BUDGET_SECONDS,
    request_timeout_seconds: Annotated[
        float, typer.Option("--request-timeout-seconds")
    ] = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    kaggle_account: Annotated[
        str | None,
        typer.Option(
            "--kaggle-account",
            help=(
                "Kaggle account profile accN, or 'auto' to choose the account "
                "with the most GPU quota before every session."
            ),
        ),
    ] = None,
    max_runs: Annotated[
        int | None,
        typer.Option(
            "--max-runs",
            min=1,
            help="Stop after this many Kaggle sessions (default: until complete "
            "or out of quota).",
        ),
    ] = None,
) -> None:
    adapter = (
        LocalRerankBackend() if backend is Backend.LOCAL else KaggleRerankBackend()
    )
    request = RerankRequest(
        run_root=run_dir(run),
        model=model,
        force=force,
        dry_run=dry_run,
        budget_seconds=budget_seconds,
        request_timeout_seconds=request_timeout_seconds,
        kaggle_account=kaggle_account,
        max_runs=max_runs,
    )
    run_handler(state_from_context(ctx), lambda: _logged(backend, request, adapter.run))


def _logged(
    backend: Backend,
    request: RerankRequest,
    run: Callable[[RerankRequest], RerankStageResult],
) -> CommandResult:
    log = open_rerank_log(request.model, echo=False)
    log(
        f"command backend={backend.value} run={request.run_root.name} "
        f"model={request.model} kaggle_account={request.kaggle_account or 'default'} "
        f"max_runs={request.max_runs or 'unlimited'} force={request.force} "
        f"dry_run={request.dry_run}"
    )
    try:
        result = run(request)
    except BaseException as error:
        log(f"command error={type(error).__name__}: {error}")
        raise
    status = "incomplete" if result.incomplete else "complete"
    log(f"command status={status} actions={' | '.join(result.actions)}")
    return _result(result)


def _result(result: RerankStageResult) -> CommandResult:
    return CommandResult(
        "rerank",
        CommandStatus.INCOMPLETE if result.incomplete else CommandStatus.COMPLETE,
        result.artifact_dir,
        {
            "actions": result.actions,
            "subset_sha256": result.subset_sha256,
            "variant_sha256": result.variant_sha256,
            "benchmark_report": result.benchmark_report,
            "benchmark_levels": result.benchmark_levels,
            "quota": result.quota,
        },
    )
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest -q tests/evaluation/test_rerank_service.py tests/cli/test_rerank_command.py`
Expected: PASS.

- [ ] **Step 7: Update the guides**

In `seed-pipeline/docs/guides/workflow-local-kaggle.md` replace section `## 6. Rerank trên Kaggle và metrics` (heading to the line before `## Troubleshooting`) with:

````markdown
## 6. Rerank trên Kaggle và metrics

Một model có thể cần nhiều phiên GPU 6 giờ, nên mỗi model chạy trong một cửa sổ tmux riêng; tắt terminal hay mất kết nối không làm dừng lệnh:

```bash
tmux new-session -s rerank -n qwen3-reranker-4b
uv run seed rerank --run backend-hybrid-qwen4b-p50-k30-rrf2 --backend kaggle \
  --model qwen3-reranker:4b-fp16 --kaggle-account auto --dry-run
uv run seed rerank --run backend-hybrid-qwen4b-p50-k30-rrf2 --backend kaggle \
  --model qwen3-reranker:4b-fp16 --kaggle-account auto
uv run seed metrics --run backend-hybrid-qwen4b-p50-k30-rrf2 --model qwen3-reranker:4b-fp16 --top-k 30
```

Với `--kaggle-account auto`, trước mỗi phiên lệnh:

- chạy `kaggle quota -v` bằng thông tin của từng profile `accN` trong `.env` và đọc dòng `GPU`;
- bỏ tài khoản đang bị lệnh khác khoá (`data/work/locks/kaggle-accounts/<accN>.lock`), chọn tài khoản còn nhiều giờ GPU nhất, bằng nhau thì chọn `accN` nhỏ hơn;
- đặt ngân sách phiên = min(`--budget-seconds`, giờ còn lại − 0,5 giờ) và chỉ nộp kernel khi ngân sách ít nhất 1 giờ.

Sau mỗi phiên, điểm (kể cả khi kernel dừng giữa chừng) được gộp vào `data/cache/rerank_scores/<model>.jsonl` trước, rồi mới publish checkpoint. Trước phiên kế tiếp, nguồn có nhiều cặp nhất trong cache ở máy và checkpoint của mọi tài khoản được publish làm checkpoint của tài khoản sắp chạy, nên đổi tài khoản không mất điểm. Lệnh chạy tới khi đủ cặp hoặc hết quota; `--max-runs N` giới hạn số phiên. `--kaggle-account accN` chỉ dùng một tài khoản, với cùng quy tắc quota. Dataset model và input chưa có luôn được tạo bằng profile của `KAGGLE_SHARED_OWNER` (acc1), vì vậy `.env` phải có profile đó.

Nhiều model chạy song song, mỗi model một cửa sổ tmux (`tmux new-window -n qwen3-reranker-8b`); khoá tài khoản bảo đảm mỗi tài khoản chỉ chạy một phiên GPU.

Khi không tài khoản nào còn đủ 1 giờ, lệnh dừng với exit code 3, `stop=quota-exhausted`, và in bảng quota:

```text
account  username                  remaining    total  refresh_at                locked
acc1     account_goc                   0.40h   30.00h  2026-09-19T00:00:00       no
acc2     account_phu_2                 0.20h   30.00h  2026-09-19T00:00:00       yes
```

`refresh_at` là lúc Kaggle làm mới quota của tài khoản; `locked=yes` là tài khoản đang chạy phiên của lệnh khác. Chạy lại đúng lệnh sau thời điểm đó để chấm tiếp.

Mỗi lệnh `seed rerank` ghi thêm vào `data/work/logs/rerank/<model>.log` (ví dụ `qwen3_reranker_4b_fp16.log`), có timestamp: tài khoản được chọn, quota trước và sau phiên, kernel, số cặp, lỗi và lý do dừng. Log nằm ngoài `/tmp` nên còn sau khi máy khởi động lại. Theo dõi từ cửa sổ khác:

```bash
tail -f data/work/logs/rerank/qwen3_reranker_4b_fp16.log
```

Kaggle không truy cập Postgres/Qdrant và không tính metrics; rerank chỉ nhận candidate JSONL cùng manifest.
````

In the same file, append to the `## Troubleshooting` list:

```markdown
- `stop=quota-exhausted`: xem bảng quota và chạy lại sau `refresh_at`.
- `stop=no-progress`: phiên vừa rồi không thêm cặp nào (kernel lỗi không có output hoặc llama-server không lên sau 3 lần khởi động lại); đọc `data/work/logs/rerank/<model>.log` và `server-*.log` trong output kernel.
- Máy hoặc WSL khởi động lại khi kernel đang chạy: mở lại tmux và chạy lại đúng lệnh; lệnh nối vào kernel đang chạy trên tài khoản đó thay vì nộp kernel mới.
```

In `seed-pipeline/docs/guides/workflow-local-only.md`, directly below the paragraph that starts with `Mỗi baseline cần ít nhất 30 candidates/query`, add:

```markdown
Rerank trên CPU có thể chạy nhiều giờ: chạy lệnh trong một cửa sổ tmux riêng (`tmux new-session -s rerank`). Mỗi lệnh `seed rerank` ghi dòng bắt đầu, kết quả và lỗi vào `data/work/logs/rerank/<model>.log`; file này còn sau khi máy khởi động lại.
```

In `seed-pipeline/docs/guides/cli-reference.md`:
- replace the synopsis line `seed rerank --run NAME --backend local|kaggle --model MODEL` with `seed rerank --run NAME --backend local|kaggle --model MODEL [--kaggle-account accN|auto] [--max-runs N]`;
- replace the example line `uv run seed rerank --run backend-hybrid-qwen4b-p50-k30-rrf2 --backend kaggle --model qwen3-reranker:4b-fp16 --kaggle-account acc2` with `uv run seed rerank --run backend-hybrid-qwen4b-p50-k30-rrf2 --backend kaggle --model qwen3-reranker:4b-fp16 --kaggle-account auto`;
- directly below the paragraph that starts with `` `seed rerank` chỉ đọc candidates của run`` add:

```markdown
Với `--backend kaggle`, job chạy thành nhiều phiên GPU. `--kaggle-account auto` đọc `kaggle quota -v` của mọi profile trước mỗi phiên và chọn tài khoản còn nhiều giờ nhất (bằng nhau thì `accN` nhỏ hơn) mà không bị lệnh khác khoá; `--kaggle-account accN` giữ một tài khoản. Ngân sách phiên = min(`--budget-seconds`, quota còn lại − 0,5 giờ), tối thiểu 1 giờ nên `--budget-seconds` phải ≥ 3600. Lệnh dừng khi đủ cặp (`stop=complete`), hết quota (`stop=quota-exhausted`, exit 3, in bảng quota với `refresh_at`), một phiên không thêm cặp (`stop=no-progress`) hoặc đạt `--max-runs N` (`stop=max-runs`). Điểm của mọi phiên được gộp vào cache ở máy trước khi publish checkpoint, và cache đủ cặp thì lệnh ghi biến thể mà không gọi Kaggle. `--force` chấm lại từ đầu: xoá các cặp của run khỏi cache ở máy, phiên đầu bỏ qua checkpoint, kernel cũ và publish lại dataset, các phiên sau chỉ tiếp tục từ điểm của lần chạy này. Mỗi lệnh ghi log có timestamp vào `data/work/logs/rerank/<model>.log`.
```

- [ ] **Step 8: Run the seed gate**

Run the **Seed gate**. Expected: all green.

- [ ] **Step 9: Commit**

```bash
git add seed-pipeline/src/seed_pipeline/evaluation/rerank_service.py \
  seed-pipeline/src/seed_pipeline/cli/commands/rerank.py \
  seed-pipeline/tests/evaluation/test_rerank_service.py \
  seed-pipeline/tests/cli/test_rerank_command.py \
  seed-pipeline/docs/guides/workflow-local-kaggle.md \
  seed-pipeline/docs/guides/workflow-local-only.md \
  seed-pipeline/docs/guides/cli-reference.md
git diff --cached --name-status
git commit -m "feat(seed): rerank on Kaggle with quota-aware sessions across accounts

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Recover the scores of a finished kernel

**Files:**
- Modify: `seed-pipeline/src/seed_pipeline/evaluation/rerank_service.py`
- Modify: `seed-pipeline/src/seed_pipeline/cli/commands/rerank.py`
- Modify: `seed-pipeline/docs/guides/cli-reference.md`, `seed-pipeline/docs/guides/workflow-local-kaggle.md`
- Test: `seed-pipeline/tests/evaluation/test_rerank_service.py`, `seed-pipeline/tests/cli/test_rerank_command.py`

**Interfaces:**
- Consumes: `resolve_profile_execution_contexts()` (service), `KernelService(runner, owner)` with `inspect_state(reference) -> KernelRemoteState` and `download_output(reference, destination) -> None` (kernel_service), `managed_staging_directory(parent, *, prefix)`, `sha256_file(path)` (integrations.kaggle.artifacts), `_merge_remote_rerank_scores`, `_local_subset`, `_finalize_variant` (Task 9).
- Produces:
  - `RerankRequest.recover_kernel: str | None = None`.
  - `KaggleRerankBackend._recover_kernel(request: RerankRequest) -> RerankStageResult` — runs under the model's job lock; finds the profile whose username is the kernel owner; refuses a kernel that is not `COMPLETE` or `ERROR`; downloads its output into `data/work/kaggle-rerank-recovery/`; checks that exactly one `rerank_scores.jsonl` has a `manifest.json` with `artifact_type == "rerank_scores"`, `identity.model == --model` and a matching `data_sha256`; merges the records into the local cache (the cache rejects records whose model or request-contract digest differs from the catalog); returns actions `kernel=<ref>`, `recovered_pairs=N`, `missing_pairs=N` and registers the variant when the run's pairs are complete.
  - Module-level helper `rerank_service._recovered_scores(root: Path, model: str) -> Path` (raises `ValueError` for a missing, foreign-model or corrupted score file).
  - CLI: `seed rerank --backend kaggle --recover-kernel OWNER/KERNEL-SLUG`; `LocalRerankBackend` raises `ValueError("--recover-kernel requires --backend kaggle")`.

Why: kernel `doanvanan0209/rerank-5f22fcadeede1072` finished with 94,950 scored pairs that never reached the local cache. After Plan A its job identity (runtime profile, `reuse_sha256`) differs from any new job, so neither the reconcile path nor the checkpoint slugs find it; records are matched by `(reranker, model_sha256, request_contract_sha256, query_id, query_hash, chunk_id, document_hash)` instead. Plan C runs, in tmux:

```bash
uv run seed rerank --run hybrid-qwen4b-p50-k30-rrf2 --backend kaggle \
  --model qwen3-reranker:0.6b-fp16 --recover-kernel doanvanan0209/rerank-5f22fcadeede1072
```

The merge relies on Plan A's frozen `native_rerank_contract().sha256` (`95b81f733a6695906ec4b9c0a30ab9588dc1f43bd9e64e45800101c87ee0eb48`) and an unchanged model sha256 for `qwen3-reranker:0.6b-fp16`; if either differs, the cache raises `Prompt contract mismatch` or `Model digest mismatch` and nothing is merged.

- [ ] **Step 1: Write the failing backend tests**

In `seed-pipeline/tests/evaluation/test_rerank_service.py` add the imports (then `uv run ruff check --fix tests/evaluation/test_rerank_service.py`):

```python
import shutil
from collections.abc import Callable
from pathlib import Path

from seed_pipeline.integrations.kaggle import kernel_service
from seed_pipeline.integrations.kaggle.models import (
    KernelPresence,
    KernelRemoteState,
    KernelStatus,
)
from seed_pipeline.integrations.kaggle.workers.runtime import write_artifact_manifest
```

(merge the models names into the existing models import) and append:

```python
KERNEL = "user-acc2/rerank-5f22fcadeede1072"


def _fake_kernel_service(
    status: KernelStatus,
    write_output: Callable[[Path], None],
    seen: list[tuple[str, str]],
):
    class FakeKernelService:
        def __init__(self, runner, owner):
            self.owner = owner

        def inspect_state(self, reference):
            seen.append(("inspect", reference))
            return KernelRemoteState(reference, KernelPresence.EXISTS, status)

        def download_output(self, reference, destination):
            seen.append(("download", self.owner))
            write_output(Path(destination))

    return FakeKernelService


def _kernel_output(cache: RerankScoreCache, *, model: str = MODEL):
    def write_output(destination: Path) -> None:
        artifact_dir = destination / "artifact"
        artifact_dir.mkdir(parents=True)
        data = artifact_dir / "rerank_scores.jsonl"
        shutil.copy2(cache.path, data)
        write_artifact_manifest(
            data,
            artifact_type="rerank_scores",
            identity=job_identity(model=model),
            completion=Completion(300_000, 1, 299_999),
        )

    return write_output


@pytest.fixture
def recovery_accounts(monkeypatch, tmp_path):
    monkeypatch.setattr(rerank_service, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(
        kaggle_service,
        "resolve_profile_execution_contexts",
        lambda: _session_contexts("acc1", "acc2"),
    )


def test_recover_kernel_merges_its_scores_and_registers_a_complete_variant(
    complete_run, complete_rerank_cache, recovery_accounts, monkeypatch
):
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(
        kernel_service,
        "KernelService",
        _fake_kernel_service(
            KernelStatus.ERROR, _kernel_output(complete_rerank_cache), seen
        ),
    )

    result = KaggleRerankBackend().run(
        replace(request(complete_run, MODEL), recover_kernel=KERNEL)
    )

    assert seen == [("inspect", KERNEL), ("download", "user-acc2")]
    assert "recovered_pairs=1" in result.actions
    assert "missing_pairs=0" in result.actions
    assert result.artifact_dir == complete_run / "rerank" / "qwen3_reranker_0_6b_fp16"


def test_recover_kernel_refuses_a_running_kernel(
    complete_run, complete_rerank_cache, recovery_accounts, monkeypatch
):
    monkeypatch.setattr(
        kernel_service,
        "KernelService",
        _fake_kernel_service(
            KernelStatus.RUNNING, _kernel_output(complete_rerank_cache), []
        ),
    )

    with pytest.raises(ValueError, match="no finished output"):
        KaggleRerankBackend().run(
            replace(request(complete_run, MODEL), recover_kernel=KERNEL)
        )


def test_recover_kernel_rejects_scores_of_another_model(
    complete_run, complete_rerank_cache, recovery_accounts, monkeypatch
):
    monkeypatch.setattr(
        kernel_service,
        "KernelService",
        _fake_kernel_service(
            KernelStatus.COMPLETE,
            _kernel_output(complete_rerank_cache, model="bge-reranker-v2-m3:f16"),
            [],
        ),
    )

    with pytest.raises(ValueError, match="another model"):
        KaggleRerankBackend().run(
            replace(request(complete_run, MODEL), recover_kernel=KERNEL)
        )
    assert load_run_record(complete_run / "run.json").rerank_variants == {}


def test_recover_kernel_needs_a_profile_for_the_kernel_owner(
    complete_run, recovery_accounts
):
    with pytest.raises(ValueError, match="has username stranger"):
        KaggleRerankBackend().run(
            replace(request(complete_run, MODEL), recover_kernel="stranger/rerank-1")
        )


def test_local_backend_rejects_kernel_recovery(complete_run):
    with pytest.raises(ValueError, match="requires --backend kaggle"):
        fake_local_backend().run(
            replace(request(complete_run, MODEL), recover_kernel=KERNEL)
        )
```

- [ ] **Step 2: Write the failing CLI test**

Append to `seed-pipeline/tests/cli/test_rerank_command.py`:

```python
def test_rerank_passes_the_kernel_to_recover(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        rerank_command, "KaggleRerankBackend", fake_backend(captured, incomplete=True)
    )

    result = runner.invoke(
        app,
        [
            "rerank",
            "--run",
            "experiment",
            "--backend",
            "kaggle",
            "--recover-kernel",
            "doanvanan0209/rerank-5f22fcadeede1072",
        ],
    )

    assert result.exit_code == 3
    assert captured["request"].recover_kernel == "doanvanan0209/rerank-5f22fcadeede1072"
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest -q tests/evaluation/test_rerank_service.py tests/cli/test_rerank_command.py`
Expected: FAIL — `TypeError: ... unexpected keyword argument 'recover_kernel'` and `No such option: --recover-kernel`.

- [ ] **Step 4: Implement recovery**

In `seed-pipeline/src/seed_pipeline/evaluation/rerank_service.py`:

Add `import json` to the imports and `from seed_pipeline.integrations.kaggle.artifacts import sha256_file`.

Add `recover_kernel: str | None = None` as the last field of `RerankRequest`.

At the top of `LocalRerankBackend.run`, before `spec = require_model(request.model)`, add:

```python
        if request.recover_kernel is not None:
            raise ValueError("--recover-kernel requires --backend kaggle")
```

Add below `_finalize_variant`:

```python
def _recovered_scores(root: Path, model: str) -> Path:
    """The verified rerank score file inside a downloaded kernel output."""
    manifests = [
        path
        for path in Path(root).rglob("manifest.json")
        if (path.parent / "rerank_scores.jsonl").is_file()
    ]
    if len(manifests) != 1:
        raise ValueError(
            "Expected one rerank_scores.jsonl with a manifest.json in the kernel "
            f"output, found {len(manifests)}"
        )
    manifest_path = manifests[0]
    data_path = manifest_path.parent / "rerank_scores.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("artifact_type") != "rerank_scores"
        or manifest.get("data_filename") != data_path.name
    ):
        raise ValueError(
            f"Kernel output is not a rerank score artifact: {manifest_path}"
        )
    identity = manifest.get("identity")
    if not isinstance(identity, dict) or identity.get("model") != model:
        raise ValueError(
            f"Kernel output holds scores of another model than {model}: {manifest_path}"
        )
    if manifest.get("data_sha256") != sha256_file(data_path):
        raise ValueError(
            f"Kernel output checksum does not match its manifest: {data_path}"
        )
    return data_path
```

Replace `KaggleRerankBackend._run_kaggle` with:

```python
    @staticmethod
    def _run_kaggle(request: RerankRequest) -> RerankStageResult:
        spec = require_model(request.model)
        # Kaggle staging/runtime resources are model-scoped under WORK_DIR,
        # so serialize the same model across runs while allowing variants
        # for different models to proceed concurrently.
        lock_target = WORK_DIR / "kaggle-rerank-jobs" / spec.slug
        with kaggle_job_lock(lock_target):
            if request.recover_kernel is not None:
                return KaggleRerankBackend._recover_kernel(request)
            return KaggleRerankBackend._run_kaggle_unlocked(request)
```

Add to `KaggleRerankBackend`:

```python
    @staticmethod
    def _recover_kernel(request: RerankRequest) -> RerankStageResult:
        """Merge the scores a finished kernel left in its output into the local cache.

        Works across job identities: every record carries the model and request
        contract digests and the query and document hashes, and the local cache
        rejects records that do not match the current catalog entry.
        """
        from seed_pipeline.integrations.kaggle.kernel_service import KernelService
        from seed_pipeline.integrations.kaggle.models import (
            KernelPresence,
            KernelStatus,
        )
        from seed_pipeline.integrations.kaggle.service import (
            resolve_profile_execution_contexts,
        )
        from seed_pipeline.integrations.kaggle.workspace import (
            managed_staging_directory,
        )

        reference = str(request.recover_kernel)
        owner, separator, slug = reference.partition("/")
        if not separator or not owner or not slug:
            raise ValueError("--recover-kernel must be OWNER/KERNEL-SLUG")
        spec = require_model(request.model)
        if spec.kind is not ModelKind.RERANKER:
            raise ValueError("--model must select a reranker model")
        if spec.rerank_contract is None:
            raise ValueError(f"Reranker {request.model} has no scoring contract")
        contract = spec.rerank_contract.sha256
        workspace = _workspace(request)
        candidate_bundle = load_bundle(
            workspace.candidates_dir,
            expected_type="retrieval_candidates",
            require_complete=True,
        )
        identity = RerankVariantIdentity.create(
            candidate_bundle.manifest.data_sha256, request.model
        )
        context = next(
            (
                item
                for item in resolve_profile_execution_contexts()
                if item.profile is not None
                and item.profile.username.casefold() == owner.casefold()
            ),
            None,
        )
        if context is None:
            raise ValueError(
                f"No Kaggle account profile in seed-pipeline/.env has username {owner}"
            )
        kernels = KernelService(context.runner, context.owners.execution)
        remote = kernels.inspect_state(reference)
        if remote.presence is not KernelPresence.EXISTS or remote.status not in {
            KernelStatus.COMPLETE,
            KernelStatus.ERROR,
        }:
            raise ValueError(
                f"Kernel {reference} has no finished output to recover "
                f"(presence={remote.presence.value}, status={remote.status})"
            )
        log = open_rerank_log(request.model, echo=True)
        cache_path = rerank_score_cache_path(request.model)
        with managed_staging_directory(
            WORK_DIR / "kaggle-rerank-recovery", prefix=f"{spec.slug}-"
        ) as staging:
            kernels.download_output(reference, staging)
            remote_scores = RerankScoreCache(
                _recovered_scores(staging, request.model),
                model_sha256=spec.sha256,
                request_contract_sha256=contract,
            )
            with kaggle_cache_lock(cache_path):
                _merge_remote_rerank_scores(
                    cache_path,
                    remote_scores,
                    model_sha256=spec.sha256,
                    request_contract_sha256=contract,
                )
        subset = _local_subset(
            cache_path,
            candidate_bundle.data_path,
            request.model,
            model_sha256=spec.sha256,
            request_contract_sha256=contract,
        )
        recovered = f"recovered_pairs={len(remote_scores.records)}"
        log(
            f"recovered kernel={reference} {recovered} "
            f"local cache pairs={subset.complete}/{subset.total}"
        )
        actions = (f"kernel={reference}", recovered, f"missing_pairs={subset.missing}")
        if not subset.is_complete:
            return RerankStageResult(None, identity.sha256, None, actions, incomplete=True)
        existing = _existing_variant(
            workspace, spec.slug, identity.sha256, force=request.force
        )
        if existing is not None:
            bundle = load_registered_rerank_bundle(workspace, existing)
            return RerankStageResult(
                bundle.root, identity.sha256, bundle.manifest.data_sha256, actions
            )
        return _finalize_variant(
            request, workspace, candidate_bundle, cache_path, identity, subset, actions
        )
```

- [ ] **Step 5: Add the CLI option**

In `seed-pipeline/src/seed_pipeline/cli/commands/rerank.py` add the parameter after `max_runs`:

```python
recover_kernel: Annotated[
    str | None,
    typer.Option(
        "--recover-kernel",
        help="OWNER/KERNEL-SLUG of a finished Kaggle kernel whose scores are "
        "merged into the local score cache (kaggle backend).",
    ),
] = (None,)
```

and pass `recover_kernel=recover_kernel,` to `RerankRequest(...)`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest -q tests/evaluation/test_rerank_service.py tests/cli/test_rerank_command.py`
Expected: PASS.

- [ ] **Step 7: Update the guides**

In `seed-pipeline/docs/guides/cli-reference.md` replace the synopsis line `seed rerank --run NAME --backend local|kaggle --model MODEL [--kaggle-account accN|auto] [--max-runs N]` with `seed rerank --run NAME --backend local|kaggle --model MODEL [--kaggle-account accN|auto] [--max-runs N] [--recover-kernel OWNER/SLUG]`, and below the paragraph added in Task 9 add:

```markdown
`--recover-kernel OWNER/SLUG` (chỉ với `--backend kaggle`) tải output của một kernel đã kết thúc bằng profile có username `OWNER`, kiểm manifest (`rerank_scores`, đúng `--model`, đúng sha256) rồi gộp điểm vào `data/cache/rerank_scores/<model>.jsonl`, không nộp kernel mới. Điểm được khớp theo model, request contract, query và tài liệu, nên cách này dùng được cả khi job identity đã đổi. Khi cache đủ cặp của run, lệnh ghi luôn biến thể.
```

In `seed-pipeline/docs/guides/workflow-local-kaggle.md` append to the `## Troubleshooting` list:

```markdown
- Kernel đã kết thúc nhưng điểm chưa về máy và lệnh không còn nối được vào nó (ví dụ cấu hình runtime đã đổi nên job identity khác): `uv run seed rerank --run RUN --backend kaggle --model MODEL --recover-kernel OWNER/SLUG`, rồi chạy lại lệnh rerank thường để chấm phần còn thiếu.
```

- [ ] **Step 8: Run the seed gate**

Run the **Seed gate**. Expected: all green.

- [ ] **Step 9: Commit**

```bash
git add seed-pipeline/src/seed_pipeline/evaluation/rerank_service.py \
  seed-pipeline/src/seed_pipeline/cli/commands/rerank.py \
  seed-pipeline/tests/evaluation/test_rerank_service.py \
  seed-pipeline/tests/cli/test_rerank_command.py \
  seed-pipeline/docs/guides/cli-reference.md \
  seed-pipeline/docs/guides/workflow-local-kaggle.md
git diff --cached --name-status
git commit -m "feat(seed): recover the scores of a finished Kaggle rerank kernel

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Spec Coverage

| Spec item | Task |
| --- | --- |
| 4.8 server crash: restart llama-server up to 3 times, wait for `/health`, score missing pairs, then seal partial | 4 |
| 4.8 job out of budget: partial artifact merged into cache, re-run resumes | 7 (sink), 9 |
| 4.8 persistent log `data/work/logs/rerank/<model-slug>.log` with account, quota before/after, kernel, progress, errors | 1, 8, 9 |
| 4.8 long-running commands in tmux; re-run attaches to the running kernel | 7 (attach counts as a run), 8 (`active_kernel_profile`, preferred account), 9 (docs) |
| 4.9 `--kaggle-account auto`; explicit `accN` still works | 8 (`resolve_session_contexts`), 9 |
| 4.9 quota read per profile, skip locked, most remaining, tie lowest N, budget = min(budget, remaining − 0.5 h), ≥ 1 h | 2, 8 |
| 4.9 no account qualifies: `incomplete` with quota table and `refreshAt` | 2 (`quota_table`), 8, 9 |
| 4.9 account lock `data/work/locks/kaggle-accounts/<accN>.lock` held for the session | 3, 8 |
| 4.9 missing dependency datasets created with `KAGGLE_SHARED_OWNER` context | 6 |
| 4.9 merge artifact locally before checkpoint publish; best of {local cache, all account checkpoints} published before every session | 5, 7, 9 |
| 4.9 loop until complete or out of quota; `--max-runs` optional | 8, 9 |
| 6 tests: account selection, locks, cross-account checkpoints, dependencies, worker restarts, log | 2, 3, 5, 6, 4, 1/9 |
| 7 docs: `workflow-local-kaggle.md`, `workflow-local-only.md`, `cli-reference.md`, `data/README.md` | 9, 10, 1 |
| 8 risks: quota lag (re-read before every session, 0.5 h margin, Kaggle-cut sessions merged like budget exhaustion); concurrent sessions per account (lock) | 2, 7, 8, 3 |
