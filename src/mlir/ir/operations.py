from pydantic import BaseModel, NonNegativeInt, Field, model_validator
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
    easily traverse the IR. Model validators are used to ensure that the use is recorded
    in the value.
    """

    owner: "Operation"
    """The operation that owns this operand."""

    value: Value[T]
    """The value of the operand, which is an instance of Value."""

    index: NonNegativeInt
    """The index of the operand in the operation's operands list."""

    @model_validator("value", mode="before")
    def in_value_uses(self):
        """Ensures that this operand is in the uses of the value it represents."""
        if self not in self.value.uses:
            self.value.add_use(self)
        return self


class Operation(BaseModel):
    """Base class for MLIR operations.

    Contains the data the makes up an operations. Data structures are assembled using
    Pydantic to ensure validity.

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

    @model_validator(mode="after")
    def validate_operands(self):
        """Ensures that each operand's owner is this operation, and that the index is
        correct. If the owner is not set, then this will set it to this operation."""

        for idx, operand in enumerate(self.operands):
            if operand.owner != self and operand.owner is not None:
                raise ValueError(
                    f"Operand {operand} does not belong to operation {self}."
                )
            elif operand.owner is None:
                operand.owner = self

            if operand.index != idx:
                raise ValueError(
                    f"Operand {operand} has index {operand.index}, but should be {idx}."
                )
        return self

    @model_validator(mode="after")
    def validate_results(self):
        """Ensures that each result's owner is this operation, and that the index is
        correct. If the owner is not set, then this will set it to this operation."""

        for idx, result in enumerate(self.results):
            if result.owner != self and result.owner is not None:
                raise ValueError(
                    f"Result {result} does not belong to operation {self}."
                )
            elif result.owner is None:
                result.owner = self

            if result.index != idx:
                raise ValueError(
                    f"Result {result} has index {result.index}, but should be {idx}."
                )
        return self

    @model_validator(mode="after")
    def validate_regions(self):
        """Ensures that each region's parent operation is this operation. If the parent
        operation is not set, then this will set it to this operation."""

        for region in self.regions:
            if region.parent != self and region.parent is not None:
                raise ValueError(
                    f"Region {region} does not belong to operation {self}."
                )
            elif region.parent is None:
                region.parent = self
        return self
