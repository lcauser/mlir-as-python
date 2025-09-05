import pytest

from mlir.ir.operations import Operation, OpResult
from mlir.ir.traits.results import NResults
from mlir.ir.value import BlockArgument
from mlir.ir.types.numbers import IntegerType
from mlir.utils.validator import ValidatorError


def test_cache():
    res1 = NResults(5)
    res2 = NResults(5)
    assert res1 is res2

    res3 = NResults(3)
    assert res1 is not res3
    res4 = NResults(3)
    assert res3 is res4

    res5 = NResults(5)
    assert res1 is res5


@pytest.mark.parametrize("n", [0, 1, 2, 3], scope="class")
class TestNResults:
    @pytest.fixture(scope="class")
    def op_class(self, n):
        trait = NResults(n)

        class DummyOp(trait, Operation):
            def create_results(self, operands=None, **kwargs) -> list[OpResult]:
                t = IntegerType(32)
                # Always create n results
                return [OpResult(t, self, i) for i in range(n)]

        return DummyOp

    def test_n_results_creates_op(self, op_class, n):
        t = IntegerType(32)
        op = op_class(
            operands=[BlockArgument(t, None, i) for i in range(n)], attributes={}
        )
        assert len(op.results) == n

    def test_n_results_with_less_raises(self, op_class, n):
        t = IntegerType(32)
        if n == 0:
            return

        # create_results will always create n results, so we simulate fewer results by overriding
        class FewerResultsOp(op_class):
            def create_results(self, operands=None, **kwargs):
                t = IntegerType(32)
                return [OpResult(t, self, i) for i in range(n - 1)]

        with pytest.raises(ValidatorError, match=f"requires exactly {n} results"):
            FewerResultsOp(
                operands=[BlockArgument(t, None, i) for i in range(n)], attributes={}
            )

    def test_n_results_with_more_raises(self, op_class, n):
        t = IntegerType(32)

        # create_results will always create n results, so we simulate more results by overriding
        class MoreResultsOp(op_class):
            def create_results(self, operands=None, **kwargs):
                t = IntegerType(32)
                return [OpResult(t, self, i) for i in range(n + 1)]

        with pytest.raises(ValidatorError, match=f"requires exactly {n} results"):
            MoreResultsOp(
                operands=[BlockArgument(t, None, i) for i in range(n)], attributes={}
            )
