from mlir.ir.value import BlockArgument
from mlir.ir.operations import Operation
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    Region = None  # TODO add region


class Block:
    def __init__(
        self,
        arguments: list[BlockArgument] = [],
        operations: list[Operation] = [],
        owner: "Region" | None = None,
    ):
        self._arguments = []
        self._operations = []
        self.owner = owner

        for argument in arguments:
            self.add_argument(argument)
        for operation in operations:
            self.add_operation(operation)

    @property
    def number_of_arguments(self) -> int:
        """Returns the number of arguments in the block."""
        return len(self._arguments)

    def get_argument(self, index: int) -> BlockArgument:
        """Returns the block argument at the specified index."""
        return self._arguments[index]

    def add_argument(self, argument: BlockArgument):
        """Adds a block argument to the block, setting the argument's owner to this block."""
        self.insert_argument(self.number_of_arguments, argument)

    def insert_argument(self, index: int, argument: BlockArgument):
        """Inserts a block argument at the specified index, updating the indices of
        subsequent arguments."""
        if argument.owner is not None and argument.owner != self:
            raise ValueError(
                f"BlockArgument {argument} already has an owner {argument.owner}, cannot "
                f"reassign to {self}."
            )
        argument.owner = self
        self._arguments.insert(index, argument)
        for i in range(index, self.number_of_arguments):
            self._arguments[i].index = i

    def remove_argument(self, argument: BlockArgument):
        """Removes a block argument from the block, updating the indices of subsequent
        arguments."""
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

    def add_operation(self, operation: Operation):
        """Adds an operation to the block, setting the operation's parent to this block."""
        if operation.parent is not None and operation.parent != self:
            raise ValueError(
                f"Operation {operation} already has a parent {operation.parent}, cannot "
                f"reassign to {self}."
            )
        operation.parent = self
        self._operations.append(operation)

    def __repr__(self):
        return f"Block(arguments={self._arguments}, operations={self._operations}, owner={self.owner})"
