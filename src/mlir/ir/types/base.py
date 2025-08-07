from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlir.context import MLIRContext


class TypeBase(ABC, BaseModel):
    """Base class for MLIR typing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    @classmethod
    def get(cls, context: "MLIRContext", *args):
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
    def validate(self, value):
        """Raises if the value doesn't match the type."""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Return a string representation of the type."""
        pass

    @abstractmethod
    def __hash__(self) -> int:
        """Method to hash the type for deduplication. Should just be a tuple hash of the
        parameters, but details are left to the subclass."""
        pass

    def __repr__(self) -> str:
        """Return a string representation of the type for debugging."""
        fields = self.__class__.model_fields
        return f"{self.__class__.__name__}({', '.join(f'{k}={getattr(self, k)}' for k in fields)})"


class TypeStorage:
    """A storage for types for deduplication."""

    def __init__(self):
        self._types = {}

    def get(self, key: tuple) -> TypeBase | None:
        """Return a type from the context by a key, which is a tuple of its type and
        parameters that define the type. If the type does not exist, return None."""
        return self._types.get(key, None)

    def add(self, key: tuple, value: TypeBase):
        """Add a type to the context. If the type already exists, this will raise an
        error."""
        if key in self._types:
            if self._types[key] == value:
                raise ValueError(f"Type {value} already exists in the context.")
            else:
                raise ValueError(
                    f"The {key} already exists in the context with a different value."
                )
        self._types[key] = value

    def __contains__(self, key: tuple) -> bool:
        return key in self._types

    def __getitem__(self, key: tuple) -> TypeBase:
        return self._types[key]
