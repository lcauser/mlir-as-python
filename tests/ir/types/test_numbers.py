from mlir.ir.types import IntegerType, SignednessSemantics, FloatTypeKind, FloatType
from mlir.context import MLIRContext


class TestIntegerType:
    def test_get_signed(self):
        """Test getting a signed integer type."""
        context = MLIRContext()
        int_type = IntegerType.get_signed(context, 32)
        assert int_type.bitwidth == 32
        assert int_type.signedness == SignednessSemantics.SIGNED

    def test_get_unsigned(self):
        """Test getting an unsigned integer type."""
        context = MLIRContext()
        int_type = IntegerType.get_unsigned(context, 32)
        assert int_type.bitwidth == 32
        assert int_type.signedness == SignednessSemantics.UNSIGNED


class TestFloatType:
    def test_bitwidth(self):
        """Test getting a float type with a specific bitwidth."""
        float_type = FloatType(FloatTypeKind.F32)
        assert float_type.bitwidth == 32
