import pytest

from mlir.ir.operations import Operation
from mlir.ir.traits.operands import NOperands
from mlir.ir.value import OpResult, BlockArgument
from mlir.ir.types.numbers import IntegerType
from mlir.utils.validator import ValidatorError


def test_cache():
    op1 = NOperands(5)
    op2 = NOperands(5)
    assert op1 is op2

    op3 = NOperands(3)
    assert op1 is not op3
    op4 = NOperands(3)
    assert op3 is op4

    op5 = NOperands(5)
    assert op1 is op5


@pytest.mark.parametrize("n", [0, 1, 2, 3], scope="class")
class TestNOperands:
    @pytest.fixture(scope="class")
    def op_class(self, n):
        trait = NOperands(n)

        class DummyOp(trait, Operation):
            def create_results(self, **kwargs) -> list[OpResult]:
                return []

        return DummyOp

    def test_n_operands_creates_op(self, op_class, n):
        t = IntegerType(32)
        op = op_class(
            operands=[BlockArgument(t, None, i) for i in range(n)], attributes={}
        )
        assert len(op.operands) == n

    def test_n_operands_with_less_raises(self, op_class, n):
        t = IntegerType(32)
        if n == 0:
            return
        with pytest.raises(ValidatorError, match=f"requires exactly {n} operands"):
            op_class(
                operands=[BlockArgument(t, None, i) for i in range(n - 1)],
                attributes={},
            )

    def test_n_operands_with_more_raises(self, op_class, n):
        t = IntegerType(32)
        with pytest.raises(ValidatorError, match=f"requires exactly {n} operands"):
            op_class(
                operands=[BlockArgument(t, None, i) for i in range(n + 1)],
                attributes={},
            )
