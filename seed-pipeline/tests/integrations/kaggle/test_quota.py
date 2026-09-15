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
