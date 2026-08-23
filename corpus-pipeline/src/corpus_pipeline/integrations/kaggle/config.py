from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class OwnerConfiguration:
    execution: str
    runtime: str
    corpus: str
    checkpoint: str


ACCOUNT_NAME = re.compile(r"acc([1-9][0-9]*)\Z")
ACCOUNT_KEY = re.compile(r"KAGGLE_ACC([1-9][0-9]*)_(USERNAME|API_TOKEN)\Z")


@dataclass(frozen=True)
class KaggleAccountProfile:
    name: str
    username: str
    api_token: str


def discover_account_profiles(environ: Mapping[str, str]) -> tuple[str, ...]:
    numbers = {
        int(match.group(1))
        for key in environ
        if (match := ACCOUNT_KEY.fullmatch(key)) is not None
    }
    return tuple(f"acc{number}" for number in sorted(numbers))


def resolve_account_profile(
    selector: str | None,
    environ: Mapping[str, str],
) -> KaggleAccountProfile | None:
    selected = selector or environ.get("KAGGLE_ACCOUNT_DEFAULT")
    if selected is None:
        return None
    selected = selected.strip()
    match = ACCOUNT_NAME.fullmatch(selected)
    if match is None:
        raise ValueError(
            "Invalid Kaggle account profile; use acc1, acc2, acc3, ..."
        )
    prefix = f"KAGGLE_ACC{int(match.group(1))}"
    username_key = f"{prefix}_USERNAME"
    token_key = f"{prefix}_API_TOKEN"
    username = environ.get(username_key, "").strip()
    token = environ.get(token_key, "").strip()
    missing = [
        key
        for key, value in ((username_key, username), (token_key, token))
        if not value
    ]
    if missing:
        available = ", ".join(discover_account_profiles(environ)) or "none"
        raise ValueError(
            f"Kaggle profile {selected} is incomplete: missing {', '.join(missing)}; "
            f"available profiles: {available}"
        )
    shared_owner = environ.get("KAGGLE_SHARED_OWNER", "").strip()
    if not shared_owner:
        raise ValueError("Kaggle profile mode requires KAGGLE_SHARED_OWNER")
    if selected == "acc1" and username.casefold() != shared_owner.casefold():
        raise ValueError("KAGGLE_ACC1_USERNAME must match KAGGLE_SHARED_OWNER")
    return KaggleAccountProfile(selected, username, token)


def profile_owner_configuration(
    profile: KaggleAccountProfile,
    environ: Mapping[str, str],
) -> OwnerConfiguration:
    shared_owner = environ.get("KAGGLE_SHARED_OWNER", "").strip()
    if not shared_owner:
        raise ValueError("Kaggle profile mode requires KAGGLE_SHARED_OWNER")
    runtime = environ.get("KAGGLE_RUNTIME_OWNER", "").strip() or shared_owner
    corpus = environ.get("KAGGLE_CORPUS_OWNER", "").strip() or shared_owner or runtime
    return OwnerConfiguration(
        execution=profile.username,
        runtime=runtime,
        corpus=corpus,
        checkpoint=profile.username,
    )


def profile_runner_environment(
    profile: KaggleAccountProfile,
    environ: Mapping[str, str],
) -> dict[str, str]:
    environment = {
        key: value
        for key, value in environ.items()
        if key not in {"KAGGLE_USERNAME", "KAGGLE_API_TOKEN", "KAGGLE_KEY", "KAGGLE_ACCOUNT_DEFAULT"}
        and ACCOUNT_KEY.fullmatch(key) is None
    }
    environment["KAGGLE_USERNAME"] = profile.username
    environment["KAGGLE_API_TOKEN"] = profile.api_token
    return environment


def resolve_owner_configuration(
    args,
    environ: Mapping[str, str],
    authenticated_owner: str | None = None,
) -> OwnerConfiguration:
    execution = (
        getattr(args, "owner", None)
        or environ.get("KAGGLE_EXECUTION_OWNER")
        or environ.get("KAGGLE_USERNAME")
        or authenticated_owner
    )
    if not execution:
        raise ValueError(
            "Kaggle owner is unavailable: set --owner/KAGGLE_USERNAME "
            "or authenticate with 'kaggle auth login'"
        )
    shared = (
        environ.get("KAGGLE_SHARED_OWNER")
        or environ.get("KAGGLE_DATASET_OWNER")
        or environ.get("KAGGLE_STORAGE_OWNER")
    )
    runtime = (
        getattr(args, "runtime_owner", None)
        or environ.get("KAGGLE_RUNTIME_OWNER")
        or shared
        or execution
    )
    corpus = (
        getattr(args, "corpus_owner", None)
        or environ.get("KAGGLE_CORPUS_OWNER")
        or shared
        or runtime
    )
    checkpoint = (
        getattr(args, "checkpoint_owner", None)
        or environ.get("KAGGLE_CHECKPOINT_OWNER")
        or execution
    )
    return OwnerConfiguration(execution, runtime, corpus, checkpoint)
