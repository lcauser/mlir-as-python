import pytest

from mlir.ir.module import ModuleOperation
from mlir.ir.regions import Region
from mlir.ir.types import IntegerType
from mlir.ir.value import BlockArgument
from mlir.utils.validator import ValidatorError


class TestModuleOperation:
    def test_module_operation_creation(self):
        module = ModuleOperation(operands=[], attributes={}, regions=[Region()])
        assert isinstance(module, ModuleOperation)
        assert len(module.regions) == 1
        assert len(module.operands) == 0
        assert len(module.results) == 0

    def test_build(self):
        module = ModuleOperation.build()
        assert isinstance(module, ModuleOperation)
        assert len(module.regions) == 1
        assert len(module.operands) == 0
        assert len(module.results) == 0

    def test_module_with_region_raises(self):
        with pytest.raises(ValidatorError, match="requires exactly 1 regions"):
            ModuleOperation(operands=[], attributes={}, regions=[Region(), Region()])

    def test_module_with_operands_raises(self):
        t = IntegerType(32)
        arg = BlockArgument(t, None, 0)
        with pytest.raises(ValidatorError, match="requires exactly 0 operands"):
            ModuleOperation(operands=[arg], attributes={}, regions=[Region()])
