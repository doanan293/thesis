from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from seed_pipeline.integrations.kaggle.checkpoints import (
    CheckpointService,
    CheckpointState,
)
from seed_pipeline.integrations.kaggle.models import (
    ActionVerb,
    ReconcileAction,
    StageJob,
)


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
    ):
        self.target_profile = target_profile
        self.target = target
        self.candidates = candidates
        self.temp_root = Path(temp_root)
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
    ) -> CheckpointInheritanceResult:
        best = target_state
        best_profile = self.target_profile
        with tempfile.TemporaryDirectory(
            prefix="checkpoint-inheritance-", dir=str(self.temp_root)
        ) as raw:
            root = Path(raw)
            for candidate in self.candidates:
                if candidate.profile_name == self.target_profile:
                    continue
                state = candidate.checkpoints.inspect(
                    job, download_root=root / candidate.profile_name
                )
                if state.completion.complete > best.completion.complete:
                    if state.artifact is None:
                        raise RuntimeError(
                            f"checkpoint candidate {candidate.profile_name} has progress without artifact"
                        )
                    best = state
                    best_profile = candidate.profile_name

            if best_profile == self.target_profile:
                return CheckpointInheritanceResult(target_state, ())
            if check_only:
                return CheckpointInheritanceResult(
                    target_state,
                    (
                        ReconcileAction(
                            "checkpoint",
                            self.target.reference(job),
                            ActionVerb.SYNC,
                            f"would inherit {best_profile} -> {self.target_profile}: "
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
                        f"checkpoint inherited {best_profile} -> {self.target_profile}: "
                        f"{best.completion.complete}/{best.completion.total} pairs",
                    ),
                ),
            )
