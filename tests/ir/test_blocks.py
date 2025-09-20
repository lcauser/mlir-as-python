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

    def test_insert_argument(self):
        block = self.create_block()
        arg = IntegerType(bitwidth=8)
        block.insert_argument(0, arg)
        block_arg = block.get_argument(0)
        assert isinstance(block_arg, BlockArgument)
        assert block_arg.type == arg
        assert block_arg.index == 0
        assert block_arg.owner == block

    def test_remove_argument_from_block(self):
        block = self.create_block()
        arg = IntegerType(bitwidth=8)
        block.insert_argument(1, arg)
        block_arg = block.get_argument(1)
        assert isinstance(block_arg, BlockArgument)
        assert block_arg.type == arg
        assert block_arg.index == 1
        assert block_arg.owner == block
        assert block.number_of_arguments == 3

        # remove it
        block.remove_argument(block_arg)
        assert block.number_of_arguments == 2
        for arg_idx in range(2):
            assert block.get_argument(arg_idx).type != arg
        assert block_arg.owner is None
        assert block_arg.index is None

    def test_remove_argument_from_index(self):
        block = self.create_block()
        arg = IntegerType(bitwidth=8)
        block.insert_argument(1, arg)
        block_arg = block.get_argument(1)
        assert isinstance(block_arg, BlockArgument)
        assert block_arg.type == arg
        assert block_arg.index == 1
        assert block_arg.owner == block
        assert block.number_of_arguments == 3

        # remove it
        block.remove_argument(1)
        assert block.number_of_arguments == 2
        for arg_idx in range(2):
            assert block.get_argument(arg_idx).type != arg
        assert block_arg.owner is None
        assert block_arg.index is None

        # should raise
        with pytest.raises(ValueError, match="does not belong to this block"):
            block.remove_argument(block_arg)

    def test_push_end(self):
        op = DummyOp([], {})
        assert op.parent is None
        block = self.create_block()
        block.push_end(op)
        assert block.number_of_operations == 3
        op_ref = block.get_operation(2)
        assert op_ref is op
        assert op_ref.parent is block

    def test_push_front(self):
        op = DummyOp([], {})
        assert op.parent is None
        block = self.create_block()
        block.push_front(op)
        assert block.number_of_operations == 3
        op_ref = block.get_operation(0)
        assert op_ref is op
        assert op_ref.parent is block

    def test_insert_operation(self):
        op = DummyOp([], {})
        assert op.parent is None
        block = self.create_block()
        block.insert_operation(1, op)
        assert block.number_of_operations == 3
        op_ref = block.get_operation(1)
        assert op_ref is op
        assert op_ref.parent is block

    def test_remove_operation(self):
        op = DummyOp([], {})
        block = self.create_block()
        block.insert_operation(1, op)
        assert block.number_of_operations == 3
        op_ref = block.get_operation(1)
        assert op_ref is op
        assert op_ref.parent is block

        block.remove_operation(op_ref)
        assert block.number_of_operations == 2
        for i in range(2):
            assert block.get_operation(i) is not op
        assert op_ref.parent is None

        with pytest.raises(ValueError, match="does not belong to this block"):
            block.remove_operation(op_ref)

    def test_clear(self):
        block = self.create_block()
        op_refs = [block.get_operation(i) for i in range(2)]
        assert block.number_of_operations == 2
        block.clear()
        assert block.number_of_operations == 0
        assert all([op.parent is None for op in op_refs])

    def test_splice(self):
        block = self.create_block()
        other_block = self.create_block()

        assert block.number_of_operations == 2
        assert other_block.number_of_operations == 2

        op = block.get_operation(0)
        assert op.parent is block
        block.splice(op, other_block, 0)

        assert op.parent is other_block
        assert block.number_of_operations == 1
        assert other_block.number_of_operations == 3

        with pytest.raises(ValueError, match="does not belong to this block"):
            block.splice(op, other_block, 2)
