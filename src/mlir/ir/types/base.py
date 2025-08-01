from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict

from mlir.context import MLIRContext


class TypeBase(ABC, BaseModel):
    """Base class for MLIR typing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

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

    @abstractmethod
    def __str__(self) -> str:
        """Return a string representation of the type."""
        pass

    def __repr__(self) -> str:
        """Return a string representation of the type for debugging."""
        fields = self.__class__.model_fields
        return f"{self.__class__.__name__}({', '.join(f'{k}={getattr(self, k)}' for k in fields)})"
