from abc import ABC
from pydantic import NonNegativeInt
from mlir.ir.types import TypeBase
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlir.ir.operations import Operation, OpOperand
    from mlir.ir.blocks import Block


class Value(ABC):
    """Base class for MLIR values, which are instances of TypeBase.

    In MLIR this would be an opaque wrapper around a pointer to the SSA value. Here I just
    model it as a base class and we will pass around references to the objects.
    """

    def __init__(self, type: TypeBase):
        self.type = type
        self.uses: list["OpOperand"] = []

    def add_use(self, use: "OpOperand"):
        """Adds a use for this value."""
        self.uses.append(use)

    def remove_use(self, use: "OpOperand"):
        """Removes a use for this value."""
        self.uses.remove(use)


class OpResult(Value):
    """Represents a result of an operation in MLIR.

    Linked to the operation that produces it, allowing us to easily traverse the IR.

    :param type: The type of the result, which is an instance of TypeBase.
    :param owner: The operation that owns this result.
    :param index: The index of the result in the operation's results list.
    """

    def __init__(
        self, type: TypeBase, owner: "Operation | None", index: NonNegativeInt
    ):
        super().__init__(type)
        self.owner = owner
        self.index = index

    def __repr__(self):
        return f"OpResult(type={self.type}, owner={self.owner}, index={self.index})"


class BlockArgument(Value):
    """Represents an SSA value argument for a block.

    Contains a reference to the block that owns it, allowing us to easily traverse the IR.

    :param type: The type of the argument, which is an instance of TypeBase.
    :param owner: The block that owns this argument.
    :param index: The index of the argument in the block's arguments list.
    """

    def __init__(
        self, type: TypeBase, owner: "Block | None", index: NonNegativeInt | None
    ):
        super().__init__(type)
        self.owner = owner
        self.index = index

    def __repr__(self):
        return (
            f"BlockArgument(type={self.type}, owner={self.owner}, index={self.index})"
        )
