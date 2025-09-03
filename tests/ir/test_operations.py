from mlir.ir.operations import Operation, OpResult
from mlir.ir.types.numbers import IntegerType
from mlir.ir.value import BlockArgument


class TestOperation:
    """Tests base behaviour of MLIR operations."""

    class DummyOperandsOp(Operation):
        def create_results(self, **kwargs) -> list[OpResult]:
            return []

    def test_validate_operands_adds_use_and_owner(self):
        operand = BlockArgument(IntegerType(32), None, 0)
        assert operand.uses == []
        op = self.DummyOperandsOp(operands=[operand], attributes={})
        assert len(op.operands) == 1
        assert op.operands[0].value is operand
        assert op.operands[0].owner is op
        assert op.operands[0] in op.operands[0].value.uses

    class DummyOp(Operation):
        def create_results(self, operands, **kwargs) -> list[OpResult]:
            results = []
            for i in range(len(operands)):
                type = IntegerType(bitwidth=32)
                results.append(OpResult(type, self, i))

            return results
