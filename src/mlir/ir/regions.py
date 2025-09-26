from mlir.ir.blocks import Block
from mlir.ir.operations import Operation


class Region:
    """Stores a list of blocks."""

    def __init__(self, blocks: list[Block] = [], parent: Operation | None = None):
        self._blocks = []
        self.parent = parent

        for block in blocks:
            self.push_end(block)

    @property
    def size(self) -> int:
        return len(self._blocks)

    @property
    def is_empty(self) -> bool:
        return self.size == 0

    @property
    def front(self) -> Block | None:
        if self.is_empty:
            return None
        return self._blocks[0]

    @property
    def end(self) -> Block | None:
        if self.is_empty:
            return None
        return self._blocks[self.size - 1]

    @property
    def blocks(self) -> list[Block]:
        return self._blocks

    def push_front(self, block: Block) -> None:
        """Inserts a block to the front of the region."""
        self.insert(0, block)

    def push_end(self, block: Block) -> None:
        """Inserts a block to the end of the region."""
        self.insert(self.size, block)

    def insert(self, index: int, block: Block) -> None:
        """Inserts a block at the specified index in the region."""
        if block.owner is not None:
            raise ValueError("Block is already owned by a region.")
        self._blocks.insert(index, block)
        block.owner = self

    def remove(self, block: Block):
        """Remove a block from the region."""
        self._blocks.remove(block)
        block.owner = None

    def clear(self):
        """Clear all of the blocks from the region."""
        for block in self._blocks:
            block.owner = None
        self._blocks = []
