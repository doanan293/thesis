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
