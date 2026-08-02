from pathlib import Path

import pytest

from corpus_pipeline.integrations.kaggle.dataset_service import (
    DatasetPresence,
    DatasetRemoteState,
)
from corpus_pipeline.integrations.kaggle.errors import KaggleRemoteStateError
from corpus_pipeline.integrations.kaggle.models import ActionVerb
from corpus_pipeline.integrations.kaggle.reconcile import DesiredDataset, plan_dataset


def desired(fingerprint: str = "abc"):
    return DesiredDataset(
        resource_kind="input",
        reference="owner/input",
        title="Input",
        fingerprint=fingerprint,
        manifest_filename="dependency_manifest.json",
        public=False,
        materialize=lambda root: Path(root),
    )


def state(presence, status=None, fingerprint=None):
    manifest = {"fingerprint": fingerprint} if fingerprint else None
    return DatasetRemoteState(presence, status=status, manifest=manifest)


@pytest.mark.parametrize(
    ("observed", "force", "expected"),
    [
        (state(DatasetPresence.ABSENT), False, ActionVerb.CREATE),
        (state(DatasetPresence.EXISTS, "READY", "abc"), False, ActionVerb.REUSE),
        (state(DatasetPresence.EXISTS, "READY", "old"), False, ActionVerb.UPDATE),
        (state(DatasetPresence.EXISTS, "READY"), False, ActionVerb.UPDATE),
        (state(DatasetPresence.EXISTS, "PENDING", "old"), False, ActionVerb.WAIT),
        (state(DatasetPresence.EXISTS, "READY", "abc"), True, ActionVerb.UPDATE),
    ],
)
def test_plan_dataset(observed, force, expected):
    assert plan_dataset(desired(), observed, force=force).verb is expected


def test_unknown_remote_state_is_not_created():
    with pytest.raises(KaggleRemoteStateError, match="unknown"):
        plan_dataset(desired(), state(DatasetPresence.UNKNOWN), force=False)
