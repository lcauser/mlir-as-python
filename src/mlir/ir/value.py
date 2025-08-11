from pydantic import BaseModel, Field, NonNegativeInt
from mlir.ir.types import TypeBase
from typing import Generic, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from mlir.ir.operations import Operation, OpOperand
    from mlir.ir.blocks import Block

T = TypeVar("T", bound=TypeBase)


class Value(BaseModel, Generic[T]):
    """Base class for MLIR values, which are instances of TypeBase

    The typing T is given here so we can restrict the types of operations.
    """

    type: T
    """The type of the value, which is an instance of TypeBase."""

    uses: list["OpOperand"] = Field(default_factory=list)
    """Contains the list of uses for this value."""


class OpResult(Value[T]):
    """Represents a result of an operation in MLIR."""

    owner: "Operation"
    """The operation that produces this result."""

    value: Value[T]
    """The value of the result, which is an instance of Value."""

    index: NonNegativeInt
    """The index of the result in the operation's results list."""


class BlockArgument(Value[T]):
    """Represents an SSA value argument for a block."""

    owner: "Block"
    """The block that owns this argument."""

    index: NonNegativeInt
    """The index of the argument in the block's arguments list."""
