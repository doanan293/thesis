from datetime import datetime
from pathlib import Path

import pytest
from tests.integrations.kaggle.factories import owners, stage_job

from pharma_lab.artifacts.manifest import Completion
from pharma_lab.integrations.kaggle.api import KaggleCommandRunner
from pharma_lab.integrations.kaggle.config import KaggleAccountProfile
from pharma_lab.integrations.kaggle.errors import KaggleCommandError
from pharma_lab.integrations.kaggle.job_lock import (
    KaggleAccountBusy,
    kaggle_account_lock,
)
from pharma_lab.integrations.kaggle.models import PipelineResult
from pharma_lab.integrations.kaggle.quota import AccountQuota
from pharma_lab.integrations.kaggle.service import KaggleExecutionContext
from pharma_lab.integrations.kaggle.sessions import (
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


def test_sessions_skip_an_account_with_no_free_kaggle_gpu_session(tmp_path):
    """Kaggle runs two batch GPU sessions per account and rejects the third push."""
    book = QuotaBook(acc1=27.58, acc3=30.0)
    sessions = ScriptedSessions(tmp_path, book, [10])
    rejected: list[str] = []
    log: list[str] = []

    def run_session(reserved: ReservedAccount, index: int) -> PipelineResult:
        if reserved.profile == "acc3":
            rejected.append(reserved.profile)
            raise KaggleCommandError(
                operation="kernels push",
                target="user-acc3/rerank",
                returncode=1,
                stdout=(
                    "Kernel push error: Maximum batch GPU session count of 2 reached."
                ),
                stderr="",
            )
        return sessions(reserved, index)

    result = run_account_sessions(
        _contexts("acc1", "acc3"),
        run_session,
        requested_budget_seconds=21_600,
        max_sessions=1,
        check_only=False,
        log=log.append,
        read_quota=book,
    )

    assert rejected == ["acc3"]
    assert [profile for profile, _budget in sessions.accounts] == ["acc1"]
    assert any("account=acc3 has no free Kaggle GPU session" in line for line in log)
    assert result.sessions == 1


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
