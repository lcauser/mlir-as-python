import pytest

from mlir.ir.operations import Operation, OpResult
from mlir.ir.regions import Region
from mlir.ir.traits.regions import NRegions
from mlir.utils.validator import ValidatorError


def test_cache():
    res1 = NRegions(5)
    res2 = NRegions(5)
    assert res1 is res2

    res3 = NRegions(3)
    assert res1 is not res3
    res4 = NRegions(3)
    assert res3 is res4

    res5 = NRegions(5)
    assert res1 is res5


@pytest.mark.parametrize("n", [0, 1, 2, 3], scope="class")
class TestNRegions:
    @pytest.fixture(scope="class")
    def op_class(self, n):
        trait = NRegions(n)

        class DummyOp(trait, Operation):
            def create_results(self, operands=None, **kwargs) -> list[OpResult]:
                return []

        return DummyOp

    def test_less_regions_throws_error(self, op_class, n):
        if n == 0:
            return
        with pytest.raises(ValidatorError, match="requires exactly"):
            op_class(
                operands=[], attributes=[], regions=[Region() for _ in range(n - 1)]
            )

    def test_more_regions_throws_error(self, op_class, n):
        with pytest.raises(ValidatorError, match="requires exactly"):
            op_class(
                operands=[], attributes=[], regions=[Region() for _ in range(n + 1)]
            )

    def test_exact_regions_succeeds(self, op_class, n):
        op = op_class(operands=[], attributes={}, regions=[Region() for _ in range(n)])
        assert len(op.regions) == n
