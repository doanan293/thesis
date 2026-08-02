from types import SimpleNamespace

from corpus_pipeline.integrations.kaggle.workers.runtime import worker_deadline
from corpus_pipeline.integrations.kaggle.workers.topology import server_layout
from corpus_pipeline.runtime.catalog import ModelTopology


def test_replicated_layout_binds_each_gpu():
    spec = SimpleNamespace(topology=ModelTopology.REPLICATED_2X1)
    assert [item.visible_devices for item in server_layout(spec)] == ["0", "1"]


def test_sharded_layout_exposes_both_gpus():
    spec = SimpleNamespace(topology=ModelTopology.SHARDED_1X2)
    assert [item.visible_devices for item in server_layout(spec)] == ["0,1"]


def test_worker_deadline_reserves_margin():
    assert (
        worker_deadline({"budget_seconds": 100, "budget_margin_seconds": 10}, lambda: 5)
        == 95
    )
