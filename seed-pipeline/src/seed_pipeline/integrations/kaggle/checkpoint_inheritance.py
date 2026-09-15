from __future__ import annotations

import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Protocol

from seed_pipeline.integrations.kaggle.checkpoints import (
    CheckpointService,
    CheckpointState,
)
from seed_pipeline.integrations.kaggle.models import (
    ActionVerb,
    ReconcileAction,
    StageJob,
)

LOCAL_SOURCE = "local"


class LocalCheckpointSource(Protocol):
    """Progress kept on this machine, offered as a checkpoint without a dataset."""

    def inspect(self, job: StageJob, /, *, download_root: Path) -> CheckpointState: ...


@dataclass(frozen=True)
class ProfileCheckpointService:
    profile_name: str
    checkpoints: CheckpointService


@dataclass(frozen=True)
class CheckpointInheritanceResult:
    state: CheckpointState
    actions: tuple[ReconcileAction, ...]


class CheckpointInheritanceService:
    def __init__(
        self,
        *,
        target_profile: str,
        target: CheckpointService,
        candidates: tuple[ProfileCheckpointService, ...],
        temp_root: Path,
        local: LocalCheckpointSource | None = None,
    ):
        self.target_profile = target_profile
        self.target = target
        self.candidates = candidates
        self.temp_root = Path(temp_root)
        self.local = local
        names = tuple(candidate.profile_name for candidate in candidates)
        if names != tuple(sorted(names, key=self._profile_number)):
            raise ValueError("checkpoint profiles must be in numeric order")
        if len(set(names)) != len(names):
            raise ValueError("checkpoint profiles must be unique")
        if names.count(target_profile) != 1:
            raise ValueError("target profile must occur exactly once")

    @staticmethod
    def _profile_number(name: str) -> int:
        match = re.fullmatch(r"acc([1-9][0-9]*)", name)
        if match is None:
            raise ValueError(f"invalid checkpoint profile: {name}")
        return int(match.group(1))

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
            # The local source is read first, so it wins ties against accounts.
            sources: list[tuple[str, Callable[[], CheckpointState]]] = []
            if self.local is not None:
                sources.append(
                    (
                        LOCAL_SOURCE,
                        partial(
                            self.local.inspect, job, download_root=root / LOCAL_SOURCE
                        ),
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
