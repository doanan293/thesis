from model_runtime.catalog import EMBEDDING_MODELS as MODEL_CATALOG
from model_runtime.catalog import ModelKind, ModelSpec, ModelTopology


def require_model(name: str) -> ModelSpec:
    try:
        return MODEL_CATALOG[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported embedding model: {name}") from exc


__all__ = ["MODEL_CATALOG", "ModelKind", "ModelSpec", "ModelTopology", "require_model"]
