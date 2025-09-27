from mlir.ir.operations import Operation, OpResult
from mlir.ir.regions import Region
from mlir.ir.traits.operands import ZeroOperands
from mlir.ir.traits.regions import OneRegion
from mlir.ir.traits.results import ZeroResults


class ModuleOperation(Operation, ZeroOperands, ZeroResults, OneRegion):
    """An operation that represents a module in MLIR, the top-level container for all
    other operations.

    A module can only contain a single region, and does not allow for any operands or
    produce any results.
    """

    def create_results(self, **kwargs) -> list[OpResult]:
        return []

    @classmethod
    def build(cls) -> "ModuleOperation":
        return ModuleOperation(operands=[], attributes={}, regions=[Region()])
