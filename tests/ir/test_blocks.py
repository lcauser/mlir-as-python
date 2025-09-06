import pytest

from mlir.ir.blocks import Block
from mlir.ir.value import BlockArgument, OpResult
from mlir.ir.types.numbers import IntegerType
from mlir.ir.operations import Operation


class DummyOp(Operation):
    def create_results(self, **kwargs) -> list[OpResult]:
        return []


class TestBlock:
    def create_block(self) -> Block:
        return Block(
            [IntegerType(32), IntegerType(16)],
            [DummyOp(operands=[], attributes={}), DummyOp(operands=[], attributes={})],
        )

    def test_block_arguments(self):
        block = self.create_block()
        assert block.number_of_arguments == 2
        for i in range(2):
            arg = block.get_argument(i)
            assert isinstance(arg, BlockArgument)
            assert arg.index == i
            assert arg.owner == block

    def test_block_operations(self):
        block = self.create_block()
        assert block.number_of_operations == 2
        for i in range(2):
            op = block.get_operation(i)
            assert isinstance(op, Operation)
            assert op.parent == block

    def test_front(self):
        op1 = DummyOp([], {})
        op2 = DummyOp([], {})
        block = Block([], [op1, op2])
        assert block.front == op1

    def test_back(self):
        op1 = DummyOp([], {})
        op2 = DummyOp([], {})
        block = Block([], [op1, op2])
        assert block.back == op2

    @pytest.mark.skip("MLIR-28")
    def test_terminator(self):
        pass

    @pytest.mark.skip("MLIR-28")
    def test_successors(self):
        pass

    def test_is_empty(self):
        assert Block([], []).is_empty
        assert not Block([], [DummyOp([], [])]).is_empty

    @pytest.mark.skip("MLIR-15")
    def test_parent_operation(self):
        pass

    def test_get_argument(self):
        block = self.create_block()
        bitwidths = [32, 16]
        for i in range(2):
            arg = block.get_argument(i)
            assert isinstance(arg, BlockArgument)
            assert arg.index == i
            assert isinstance(arg.type, IntegerType)
            assert arg.type.bitwidth == bitwidths[i]

    def test_add_argument(self):
        block = self.create_block()
        arg = IntegerType(bitwidth=8)
        block.add_argument(arg)
        block_arg = block.get_argument(2)
        assert isinstance(block_arg, BlockArgument)
        assert block_arg.type == arg
        assert block_arg.index == 2
        assert block_arg.owner == block
