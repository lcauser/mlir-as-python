from mlir.ir.value import BlockArgument
from mlir.ir.operations import Operation
from typing import TYPE_CHECKING
from mlir.ir.traits.terminator import Terminator
from mlir.ir.types import TypeBase

if TYPE_CHECKING:
    Region = None  # TODO add region


class Block:
    """A block contains a sequence of operations, and a list of block arguments that are
    available to those operations.

    TODO: implement double-linked list for blocks in a region?
    """

    def __init__(
        self,
        arguments: list[TypeBase] = [],
        operations: list[Operation] = [],
        owner: "Region | None" = None,
    ):
        self._arguments = []
        self._operations = []
        self.owner = owner

        for argument in arguments:
            self.add_argument(argument)
        for operation in operations:
            self.push_end(operation)

    @property
    def number_of_arguments(self) -> int:
        """Returns the number of arguments in the block."""
        return len(self._arguments)

    @property
    def number_of_operations(self) -> int:
        """Returns the number of operations in the block."""
        return len(self._operations)

    @property
    def front(self) -> Operation | None:
        """Returns the first operation in the block, or None if the block is empty."""
        if self.number_of_operations == 0:
            return None
        return self._operations[0]

    @property
    def back(self) -> Operation | None:
        """Returns the last operation in the block, or None if the block is empty."""
        if self.number_of_operations == 0:
            return None
        return self._operations[-1]

    @property
    def terminator(self) -> Terminator | None:
        """Returns the terminator operation of the block, or None if the block is empty or
        does not have a terminator."""
        if self.number_of_operations == 0:
            return None
        last_op = self._operations[-1]
        if isinstance(last_op, Terminator):
            return last_op
        return None

    @property
    def successors(self) -> list["Block"]:
        """Returns the successor blocks of the block, or an empty list if the block is
        empty or does not have a terminator."""
        terminator = self.terminator
        if terminator is None:
            return []
        return terminator.successors

    @property
    def is_empty(self) -> bool:
        """Returns True if the block has no operations."""
        return self.number_of_operations == 0

    @property
    def parent_operation(self) -> Operation | None:
        """Returns the parent operation of the block, or None if the block is not owned by
        a region or the region is not owned by an operation."""
        if self.owner is None:
            return None
        return self.owner.parent

    def get_argument(self, index: int) -> BlockArgument:
        """Returns the block argument at the specified index."""
        return self._arguments[index]

    def add_argument(self, argument: TypeBase):
        """Adds a block argument to the block, setting the argument's owner to this block."""
        self.insert_argument(self.number_of_arguments, argument)

    def insert_argument(self, index: int, argument: TypeBase):
        """Inserts a block argument at the specified index, updating the indices of
        subsequent arguments."""
        block_argument = BlockArgument(argument, self, index)
        self._arguments.insert(index, block_argument)
        for i in range(index, self.number_of_arguments):
            self._arguments[i].index = i

    def remove_argument(self, argument: BlockArgument | int):
        """Removes a block argument from the block, updating the indices of subsequent
        arguments."""
        if isinstance(argument, int):
            argument = self.get_argument(argument)
        if argument.owner != self:
            raise ValueError(
                f"BlockArgument {argument} does not belong to this block {self}, cannot "
                f"remove."
            )
        index = argument.index
        argument.owner = None
        self._arguments.remove(argument)
        for i in range(index, self.number_of_arguments):
            self._arguments[i].index = i

    def get_operation(self, index: int) -> Operation:
        """Returns the operation at the specified index."""
        return self._operations[index]

    def push_end(self, operation: Operation):
        """Adds an operation to the block, setting the operation's parent to this block."""
        return self.insert_operation(self.number_of_operations, operation)

    def push_front(self, operation: Operation):
        """Adds an operation to the front of the block, setting the operation's parent to
        this block."""
        return self.insert_operation(0, operation)

    def insert_operation(self, index: int, operation: Operation):
        """Inserts an operation at the specified index, updating the indices of subsequent
        operations."""
        if operation.parent is not None and operation.parent != self:
            raise ValueError(
                f"Operation {operation} already has a parent {operation.parent}, cannot "
                f"reassign to {self}."
            )
        operation.parent = self
        self._operations.insert(index, operation)

    def remove_operation(self, operation: Operation | int):
        """Removes an operation from the block."""
        if isinstance(operation, int):
            operation = self.get_operation(operation)
        if operation.parent != self:
            raise ValueError(
                f"Operation {operation} does not belong to this block {self}, cannot "
                f"remove."
            )
        operation.parent = None
        self._operations.remove(operation)

    def clear(self):
        """Removes all operations from the block."""
        for op in self._operations:
            op.parent = None
        self._operations.clear()

    def splice(self, operation: Operation | int, target_block: "Block", index: int):
        """Moves an operation from this block to the target block at the specified index."""
        if isinstance(operation, int):
            operation = self.get_operation(operation)
        if operation.parent != self:
            raise ValueError(
                f"Operation {operation} does not belong to this block {self}, cannot "
                f"splice."
            )
        self.remove_operation(operation)
        target_block.insert_operation(index, operation)

    def __repr__(self):
        return f"Block(arguments={self._arguments}, operations={self._operations}, owner={self.owner})"
