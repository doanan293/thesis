"""Checkpoint serializer that explicitly allowlists every type stored in the chat graph state.

LangGraph warns when it deserializes unregistered types and will block them in a future release,
so the allowlist is derived from the state schema instead of being maintained by hand.
"""

import types as pytypes
from enum import Enum
from typing import Any, Union, get_args, get_origin

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel

from pharma_agent.application.chat.state import ChatTurnState


def checkpoint_types(root: type[BaseModel] = ChatTurnState) -> list[type]:
    found: dict[type, None] = {}

    def visit(annotation: Any) -> None:
        origin = get_origin(annotation)
        if origin is not None or isinstance(annotation, pytypes.UnionType):
            for arg in get_args(annotation):
                visit(arg)
            return
        if annotation is Union or not isinstance(annotation, type):
            return
        if annotation in found:
            return
        if issubclass(annotation, Enum):
            found[annotation] = None
        elif issubclass(annotation, BaseModel):
            found[annotation] = None
            for field in annotation.model_fields.values():
                visit(field.annotation)

    visit(root)
    return list(found)


def checkpoint_serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=checkpoint_types())
