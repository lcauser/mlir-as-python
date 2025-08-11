from pydantic import BaseModel, NonNegativeInt, Field
from mlir.ir.value import Value, OpResult
from typing import Generic, TypeVar, TYPE_CHECKING
from mlir.ir.types import TypeBase
from mlir.ir.attributes import AttributeBase

if TYPE_CHECKING:
    from mlir.ir.regions import Region
    from mlir.ir.blocks import Block

T = TypeVar("T", bound=TypeBase)


class OpOperand(BaseModel, Generic[T]):
    """Represents an operand of an operation in MLIR.
    Linked to the operation that owns it and the value it represents, allowing us to
    easily traverse the IR.
    """

    owner: "Operation"
    """The operation that owns this operand."""

    value: Value[T]
    """The value of the operand, which is an instance of Value."""

    index: NonNegativeInt
    """The index of the operand in the operation's operands list."""


class Operation(BaseModel):
    """Base class for MLIR operations.

    TODO: implement as double-linked list?
    """

    operands: list[OpOperand] = Field(default_factory=list)
    """List of operands for the operation, which contain the SSA values."""

    results: list[OpResult] = Field(default_factory=list)
    """List of results for the operation, which are instances of OpResult."""

    attributes: dict[str, AttributeBase] = Field(default_factory=dict)
    """Dictionary of attributes for the operation."""

    regions: list["Region"] = Field(default_factory=list)
    """List of regions for the operation, which can nest blocks and operations."""

    parent: Block | None
    """The parent block that contains this operation, if any."""
