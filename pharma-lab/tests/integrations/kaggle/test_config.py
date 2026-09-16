import pytest

from pharma_lab.integrations.kaggle.config import (
    discover_account_profiles,
    profile_owner_configuration,
    profile_runner_environment,
    resolve_account_profile,
)


def _profiles() -> dict[str, str]:
    return {
        "PATH": "/usr/bin",
        "KAGGLE_ACCOUNT_DEFAULT": "acc1",
        "KAGGLE_SHARED_OWNER": "primary-user",
        "KAGGLE_ACC1_USERNAME": "primary-user",
        "KAGGLE_ACC1_API_TOKEN": "primary-token",
        "KAGGLE_ACC2_USERNAME": "secondary-user",
        "KAGGLE_ACC2_API_TOKEN": "secondary-token",
        "KAGGLE_ACC10_USERNAME": "tenth-user",
        "KAGGLE_ACC10_API_TOKEN": "tenth-token",
    }


def test_default_profile_resolves_acc1():
    profile = resolve_account_profile(None, _profiles())
    assert profile is not None
    assert (profile.name, profile.username, profile.api_token) == (
        "acc1",
        "primary-user",
        "primary-token",
    )


def test_explicit_profile_resolves_acc2_and_shared_owners():
    values = _profiles()
    profile = resolve_account_profile("acc2", values)
    assert profile is not None
    owners = profile_owner_configuration(profile, values)
    assert owners.execution == "secondary-user"
    assert owners.checkpoint == "secondary-user"
    assert owners.runtime == "primary-user"
    assert owners.corpus == "primary-user"


def test_profile_environment_contains_only_selected_secret():
    profile = resolve_account_profile("acc2", _profiles())
    assert profile is not None
    environment = profile_runner_environment(profile, _profiles())
    assert environment["KAGGLE_USERNAME"] == "secondary-user"
    assert environment["KAGGLE_API_TOKEN"] == "secondary-token"
    assert not any(key.startswith("KAGGLE_ACC") for key in environment)
    assert environment["PATH"] == "/usr/bin"


def test_profile_discovery_is_numeric():
    assert discover_account_profiles(_profiles()) == ("acc1", "acc2", "acc10")


def test_acc1_username_must_match_shared_owner():
    values = _profiles() | {"KAGGLE_ACC1_USERNAME": "wrong-primary"}
    with pytest.raises(ValueError, match=r"KAGGLE_ACC1_USERNAME.*KAGGLE_SHARED_OWNER"):
        resolve_account_profile("acc1", values)


@pytest.mark.parametrize("name", ["main", "gpu1", "acc0", "acc01", "ACC2", "acc2x"])
def test_invalid_profile_name_is_rejected(name):
    with pytest.raises(ValueError, match="acc1, acc2"):
        resolve_account_profile(name, _profiles())


def test_no_profile_mode_returns_none_for_legacy_environment():
    assert (
        resolve_account_profile(
            None,
            {"KAGGLE_USERNAME": "legacy", "KAGGLE_API_TOKEN": "legacy-token"},
        )
        is None
    )


@pytest.mark.parametrize(
    ("missing_key", "expected_name"),
    [
        ("KAGGLE_ACC2_USERNAME", "KAGGLE_ACC2_USERNAME"),
        ("KAGGLE_ACC2_API_TOKEN", "KAGGLE_ACC2_API_TOKEN"),
        ("KAGGLE_SHARED_OWNER", "KAGGLE_SHARED_OWNER"),
    ],
)
def test_profile_errors_name_missing_setting_without_secret(missing_key, expected_name):
    values = _profiles()
    del values[missing_key]
    with pytest.raises(ValueError) as raised:
        resolve_account_profile("acc2", values)
    message = str(raised.value)
    assert expected_name in message
    assert "primary-token" not in message
    assert "secondary-token" not in message
