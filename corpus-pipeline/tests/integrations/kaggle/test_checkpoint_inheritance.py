import pytest
from tests.integrations.kaggle.factories import cloud_artifact, stage_job

from corpus_pipeline.artifacts.manifest import Completion
from corpus_pipeline.integrations.kaggle.checkpoint_inheritance import (
    CheckpointInheritanceService,
    ProfileCheckpointService,
)
from corpus_pipeline.integrations.kaggle.checkpoints import CheckpointState
from corpus_pipeline.integrations.kaggle.models import ActionVerb


def _state(total: int, complete: int, reference: str | None = None):
    artifact = (
        cloud_artifact(Completion(total, complete, total - complete))
        if complete
        else None
    )
    return CheckpointState(
        reference, artifact, Completion(total, complete, total - complete)
    )


class FakeCheckpointService:
    def __init__(self, state, *, reference="owner/checkpoint", revalidated=None):
        self.state = state
        self.reference_value = reference
        self.revalidated = revalidated or state
        self.inspect_roots = []
        self.published = []

    def inspect(self, job, *, download_root=None):
        self.inspect_roots.append(download_root)
        return self.revalidated if self.published else self.state

    def publish_if_better(self, job, artifact, current):
        self.published.append((artifact, current))
        return CheckpointState(self.reference_value, artifact, artifact.completion)

    def reference(self, job):
        return self.reference_value


def _service(target, sources, tmp_path):
    return CheckpointInheritanceService(
        target_profile="acc3",
        target=target,
        candidates=tuple(
            ProfileCheckpointService(name, service) for name, service in sources
        ),
        temp_root=tmp_path,
    )


def test_resolve_mirrors_strictly_best_source(tmp_path):
    target = FakeCheckpointService(
        _state(300000, 0),
        reference="acc3/checkpoint",
        revalidated=_state(300000, 136336, "acc3/checkpoint"),
    )
    acc1 = FakeCheckpointService(_state(300000, 136336, "acc1/checkpoint"))
    acc2 = FakeCheckpointService(_state(300000, 120000, "acc2/checkpoint"))

    result = _service(
        target, (("acc1", acc1), ("acc2", acc2), ("acc3", target)), tmp_path
    ).resolve(stage_job(tmp_path), _state(300000, 0), check_only=False)

    assert len(target.published) == 1
    assert target.published[0][0] is acc1.state.artifact
    assert result.state.reference == "acc3/checkpoint"
    action = result.actions[0]
    assert action.verb is ActionVerb.SYNC
    assert action.resource_kind == "checkpoint"
    assert "acc1 -> acc3" in action.reason
    assert "136336/300000" in action.reason
    assert acc1.inspect_roots[0] != acc2.inspect_roots[0]


def test_resolve_keeps_target_when_tied(tmp_path):
    target = FakeCheckpointService(_state(10, 5, "acc3/checkpoint"))
    source = FakeCheckpointService(_state(10, 5, "acc1/checkpoint"))

    result = _service(target, (("acc1", source), ("acc3", target)), tmp_path).resolve(
        stage_job(tmp_path), target.state, check_only=False
    )

    assert result.state is target.state
    assert target.published == []
    assert result.actions == ()


def test_resolve_ignores_absent_and_zero_progress_sources(tmp_path):
    target = FakeCheckpointService(_state(10, 0))
    absent = FakeCheckpointService(_state(10, 0))
    zero = FakeCheckpointService(_state(10, 0))

    result = _service(
        target, (("acc1", absent), ("acc2", zero), ("acc3", target)), tmp_path
    ).resolve(stage_job(tmp_path), target.state, check_only=False)

    assert result.state is target.state
    assert target.published == []
    assert result.actions == ()


def test_resolve_uses_numeric_profile_order_for_source_tie(tmp_path):
    target = FakeCheckpointService(
        _state(10, 0), revalidated=_state(10, 4, "owner/checkpoint")
    )
    acc2 = FakeCheckpointService(_state(10, 4, "acc2/checkpoint"))
    acc10 = FakeCheckpointService(_state(10, 4, "acc10/checkpoint"))

    result = _service(
        target, (("acc2", acc2), ("acc3", target), ("acc10", acc10)), tmp_path
    ).resolve(stage_job(tmp_path), target.state, check_only=False)

    assert target.published[0][0] is acc2.state.artifact
    assert "acc2 -> acc3" in result.actions[0].reason


def test_resolve_dry_run_reports_without_publishing(tmp_path):
    target = FakeCheckpointService(_state(10, 0), reference="acc3/checkpoint")
    source = FakeCheckpointService(_state(10, 4, "acc1/checkpoint"))

    result = _service(target, (("acc1", source), ("acc3", target)), tmp_path).resolve(
        stage_job(tmp_path), target.state, check_only=True
    )

    assert target.published == []
    assert target.inspect_roots == []
    assert result.state is target.state
    assert result.actions[0].reason.startswith("would inherit")


def test_resolve_fails_closed_when_source_has_progress_without_artifact(tmp_path):
    target = FakeCheckpointService(_state(10, 0))
    source = FakeCheckpointService(
        CheckpointState("acc1/checkpoint", None, Completion(10, 4, 6))
    )

    with pytest.raises(RuntimeError, match="artifact"):
        _service(target, (("acc1", source), ("acc3", target)), tmp_path).resolve(
            stage_job(tmp_path), target.state, check_only=False
        )


def test_resolve_propagates_source_inspection_failure(tmp_path):
    target = FakeCheckpointService(_state(10, 0))

    class Broken(FakeCheckpointService):
        def inspect(self, job, *, download_root=None):
            raise RuntimeError("source inspection failed")

    source = Broken(_state(10, 4, "acc1/checkpoint"))
    with pytest.raises(RuntimeError, match="source inspection failed"):
        _service(target, (("acc1", source), ("acc3", target)), tmp_path).resolve(
            stage_job(tmp_path), target.state, check_only=False
        )
    assert target.published == []


def test_resolve_fails_when_revalidated_target_has_less_progress(tmp_path):
    target = FakeCheckpointService(
        _state(10, 0), revalidated=_state(10, 2, "owner/checkpoint")
    )
    source = FakeCheckpointService(_state(10, 4, "acc1/checkpoint"))
    with pytest.raises(RuntimeError, match="lost progress"):
        _service(target, (("acc1", source), ("acc3", target)), tmp_path).resolve(
            stage_job(tmp_path), target.state, check_only=False
        )


@pytest.mark.parametrize(
    "candidates",
    [
        (("acc1",),),
        (("acc1", "acc1"),),
        (("acc3", "acc2"),),
        (("acc3", "acc3"),),
    ],
)
def test_resolve_rejects_invalid_candidate_configuration(candidates, tmp_path):
    names = candidates[0]
    services = tuple((name, FakeCheckpointService(_state(10, 0))) for name in names)
    with pytest.raises(ValueError):
        _service(FakeCheckpointService(_state(10, 0)), services, tmp_path)
