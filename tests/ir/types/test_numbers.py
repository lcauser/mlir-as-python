from mlir.context import MLIRContext
from mlir.ir.types import FloatType, FloatTypeKind, IntegerType, SignednessSemantics


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

    def test_str(self):
        """Test the string representation of an integer type."""
        int_type = IntegerType(64)
        assert str(int_type) == "i64"


class TestFloatType:
    def test_bitwidth(self):
        """Test getting a float type with a specific bitwidth."""
        float_type = FloatType(FloatTypeKind.F32)
        assert float_type.bitwidth == 32

    def test_str(self):
        """Test the string representation of a float type."""
        float_type = FloatType(FloatTypeKind.F64)
        assert str(float_type) == "f64"
