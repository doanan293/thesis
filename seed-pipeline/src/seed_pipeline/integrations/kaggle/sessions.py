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


# Kaggle runs two batch GPU sessions per account and rejects the next kernel push.
SESSION_CAPACITY_MARKER = "maximum batch gpu session count"


class KaggleSessionCapacityReached(Exception):
    """An account already runs its two Kaggle GPU sessions."""

    def __init__(self, profile: str) -> None:
        super().__init__(f"account {profile} has no free Kaggle GPU session")
        self.profile = profile


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
    unavailable: frozenset[str] = frozenset(),
) -> SessionAccount:
    busy: set[str] = set(unavailable)
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
    unavailable: frozenset[str] = frozenset(),
) -> Generator[ReservedAccount, None, None]:
    """Read every account's GPU quota, then lock one account for one session.

    `unavailable` holds accounts a caller already found unusable, for example one whose
    Kaggle GPU sessions are all taken.
    """
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
            unavailable=unavailable,
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
    unavailable: frozenset[str] = frozenset(),
) -> PipelineResult:
    with reserve_session_account(
        contexts,
        requested_budget_seconds=requested_budget_seconds,
        log=log,
        read_quota=reader,
        account_lock=account_lock,
        preferred_profile=preferred_profile,
        unavailable=unavailable,
    ) as reserved:
        log(
            f"session={index} start account={reserved.profile} "
            f"budget_seconds={reserved.budget_seconds}"
        )
        try:
            result = run_session(reserved, index)
        except BaseException as error:
            log(f"session={index} error={type(error).__name__}: {error}")
            if SESSION_CAPACITY_MARKER in f"{error}".casefold():
                raise KaggleSessionCapacityReached(reserved.profile) from error
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
        unavailable: frozenset[str] = frozenset()
        while True:
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
                    unavailable=unavailable,
                )
            except QuotaExhausted as exhausted:
                log(f"stop=quota-exhausted sessions={sessions}; GPU quota per account:")
                for row in exhausted.table:
                    log(row)
                return SessionRunResult(
                    last, sessions, StopReason.QUOTA_EXHAUSTED, exhausted.table
                )
            except KaggleSessionCapacityReached as capped:
                log(f"account={capped.profile} has no free Kaggle GPU session; skipped")
                unavailable = unavailable | {capped.profile}
                continue
            break
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
