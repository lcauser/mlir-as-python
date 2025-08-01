from abc import ABC
from pydantic import BaseModel

from mlir.context import MLIRContext


class TypeBase(ABC, BaseModel):
    """Base class for MLIR typing."""

    class Config:
        extra = "forbid"
        frozen = True

    @classmethod
    def get(cls, context: MLIRContext, *args):
        """Get a type from the context, or create it if it does not exist."""
        key = (cls.__name__,) + args
        value = context.get_type(key)
        if value is not None:
            return value
        value = cls(*args)
        context.add_type(key, value)
        return value

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
