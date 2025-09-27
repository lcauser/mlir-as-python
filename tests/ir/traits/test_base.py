import pytest

from mlir.ir.operations import Operation
from mlir.ir.traits.base import OpTrait
from mlir.ir.types.numbers import IntegerType
from mlir.ir.value import BlockArgument, OpResult
from mlir.utils.validator import ValidatorError, validator


class TestTraitBase:
    class DummyTrait(OpTrait):
        """A trait for testing purposes - checks that the operands are integers."""

        @validator
        def validate_operand_types_are_integers(self: Operation):
            for operand in self.operands:
                if not isinstance(operand.value.type, IntegerType):
                    raise ValueError("Operand type must be IntegerType")

    class DummyOp(Operation, DummyTrait):
        def create_results(self, **kwargs) -> list[OpResult]:
            return []

    def test_trait_base_with_valid_operand_types(self):
        t = IntegerType(32)
        op = self.DummyOp(
            operands=[BlockArgument(t, None, i) for i in range(2)], attributes={}
        )
        assert len(op.operands) == 2

    def test_base_op_valiadation_runs(self):
        class DummyOpWithResults(Operation, self.DummyTrait):
            def create_results(self, **kwargs) -> list[OpResult]:
                return [OpResult(IntegerType(32), self, 1)]

        t = IntegerType(32)
        with pytest.raises(ValidatorError, match="has index"):
            DummyOpWithResults(
                operands=[BlockArgument(t, None, 0), BlockArgument(t, None, 1)],
                attributes={},
            )
