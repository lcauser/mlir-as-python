import pytest

from mlir.ir.blocks import Block
from mlir.ir.regions import Region


class TestRegion:
    def test_basic_properties(self):
        blockA = Block()
        blockB = Block()
        region = Region([blockA, blockB])
        assert region.parent is None
        assert region.size == 2
        assert region.is_empty is False
        assert region.front == blockA
        assert region.end == blockB

    def test_no_blocks(self):
        region = Region()
        assert region.parent is None
        assert region.size == 0
        assert region.is_empty is True
        assert region.front is None
        assert region.end is None

    def test_insert(self):
        blockA = Block()
        blockB = Block()
        region = Region([blockA, blockB])

        blockC = Block()
        region.insert(1, blockC)
        assert region.blocks[1] is blockC
        assert region.size == 3
        assert blockC.owner is region

        with pytest.raises(ValueError, match="Block is already owned by a region."):
            region.insert(2, blockC)

    def test_push_front(self):
        blockA = Block()
        blockB = Block()
        region = Region([blockA, blockB])

        blockC = Block()
        region.push_front(blockC)
        assert region.blocks[0] is blockC
        assert region.size == 3
        assert blockC.owner is region

    def test_push_end(self):
        blockA = Block()
        blockB = Block()
        region = Region([blockA, blockB])

        blockC = Block()
        region.push_end(blockC)
        assert region.blocks[2] is blockC
        assert region.size == 3
        assert blockC.owner is region

    def test_remove(self):
        blockA = Block()
        blockB = Block()
        region = Region([blockA, blockB])

        region.remove(blockA)
        assert region.size == 1
        assert region.blocks[0] is blockB
        assert blockA.owner is None

    def test_clear(self):
        blockA = Block()
        blockB = Block()
        region = Region([blockA, blockB])

        region.clear()
        assert region.is_empty is True
        assert region.is_empty is True
        assert blockA.owner is None
        assert blockB.owner is None
