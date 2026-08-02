from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from corpus_pipeline.integrations.kaggle.dataset_service import (
    DatasetPresence,
    DatasetRemoteState,
)
from corpus_pipeline.integrations.kaggle.errors import KaggleRemoteStateError
from corpus_pipeline.integrations.kaggle.models import ActionVerb, ReconcileAction


@dataclass(frozen=True)
class DesiredDataset:
    resource_kind: str
    reference: str
    title: str
    fingerprint: str
    manifest_filename: str
    public: bool
    materialize: Callable[[Path], Path] = field(compare=False, repr=False)


def plan_dataset(
    desired: DesiredDataset, observed: DatasetRemoteState, *, force: bool
) -> ReconcileAction:
    if observed.presence is DatasetPresence.UNKNOWN:
        raise KaggleRemoteStateError(
            f"Cannot reconcile {desired.reference}: remote state is unknown: {observed.detail}"
        )
    if observed.presence is DatasetPresence.ABSENT:
        return ReconcileAction(
            desired.resource_kind,
            desired.reference,
            ActionVerb.CREATE,
            "dataset is missing",
        )
    if observed.status != "READY":
        return ReconcileAction(
            desired.resource_kind,
            desired.reference,
            ActionVerb.WAIT,
            f"dataset status is {observed.status}",
        )
    remote_fingerprint = (observed.manifest or {}).get("fingerprint")
    if force or remote_fingerprint != desired.fingerprint:
        reason = "forced refresh" if force else "fingerprint changed"
        return ReconcileAction(
            desired.resource_kind, desired.reference, ActionVerb.UPDATE, reason
        )
    return ReconcileAction(
        desired.resource_kind,
        desired.reference,
        ActionVerb.REUSE,
        "fingerprint matches",
    )
