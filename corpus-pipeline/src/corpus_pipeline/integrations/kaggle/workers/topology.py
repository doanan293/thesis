from __future__ import annotations

from dataclasses import dataclass

from corpus_pipeline.runtime.catalog import ModelSpec, ModelTopology


@dataclass(frozen=True)
class ServerLayout:
    visible_devices: str
    tensor_split: str | None = None


def server_layout(spec: ModelSpec) -> tuple[ServerLayout, ...]:
    if spec.topology is ModelTopology.SHARDED_1X2:
        return (ServerLayout("0,1", "1,1"),)
    return (ServerLayout("0"), ServerLayout("1"))
