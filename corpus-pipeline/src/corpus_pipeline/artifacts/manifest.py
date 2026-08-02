from __future__ import annotations

from dataclasses import dataclass

from corpus_pipeline.evaluation.artifact_contracts import ArtifactContractError


@dataclass(frozen=True)
class Completion:
    """Validated progress counters shared by every artifact backend."""

    total: int
    complete: int
    missing: int

    def __post_init__(self) -> None:
        if min(self.total, self.complete, self.missing) < 0:
            raise ArtifactContractError("Completion counts must be non-negative")
        if self.complete + self.missing != self.total:
            raise ArtifactContractError(
                "Completion counts must satisfy complete + missing == total"
            )

    @property
    def is_complete(self) -> bool:
        return self.complete == self.total and self.missing == 0


ArtifactManifestError = ArtifactContractError
