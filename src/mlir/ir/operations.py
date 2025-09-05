from pydantic import NonNegativeInt
from typing import TYPE_CHECKING
from mlir.ir.attributes import AttributeBase
from mlir.ir.value import Value, OpResult
from mlir.utils.validator import validator, ValidatorMeta
from abc import ABC, abstractmethod, ABCMeta

if TYPE_CHECKING:
    from mlir.ir.blocks import Block

    # TODO: resolve with MLIR-15
    # from mlir.ir.regions import Region
    Region = list


class OpOperand:
    """Represents an operand of an operation in MLIR.

    Linked to the operation that owns it and the value it represents, allowing us to
    easily traverse the IR. Model validators are used to ensure that the use is recorded
    in the value.
    """

    def __init__(self, owner: "Operation", value: Value, index: NonNegativeInt):
        self.owner: Operation = owner
        self.value: Value = value
        self.index: NonNegativeInt = index
        if self not in self.value.uses:
            self.value.add_use(self)


class OperationMeta(ValidatorMeta, ABCMeta):
    pass


class Operation(ABC, metaclass=OperationMeta):
    """Base class for MLIR operations.

    Contains the data that makes up an operations.

    TODO: implement as double-linked list?
    """

    def __init__(
        self,
        operands: list[Value],
        attributes: dict[str, AttributeBase],
        regions: list["Region"] | None = None,
        parent: "Block | None" = None,
    ):
        self.operands: list[OpOperand] = [
            OpOperand(self, operand, idx) for idx, operand in enumerate(operands)
        ]
        self.attributes: dict[str, AttributeBase] = attributes
        self.regions: list["Region"] = regions or []
        self.parent: "Block | None" = parent
        # TODO: think about what this should be...
        self.results = self.create_results(operands=operands, attributes=attributes)

    @abstractmethod
    def create_results(self, **kwargs) -> list[OpResult]:
        """Implements a factory for creating the results list, given the operands and
        attributes. This is operation-specific logic, so it must be implemented by
        each subclass."""
        pass

    @validator
    def validate_operands(self):
        """Ensures that each operand's owner is this operation, and that the index is
        correct. If the owner is not set, then this will set it to this operation.

        # TODO: probably somewhat redundant since instantiation deals with assignment.
        """

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

    @validator
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

    @validator
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
