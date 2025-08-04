import pytest

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

    def test_str(self):
        """Test the string representation of an integer type."""
        int_type = IntegerType(64)
        assert str(int_type) == "i64"

    @pytest.mark.parametrize(
        "signedness, bitwidth, value",
        [
            (SignednessSemantics.UNSIGNED, 4, 0),
            (SignednessSemantics.UNSIGNED, 4, 15),
            (SignednessSemantics.SIGNED, 4, 7),
            (SignednessSemantics.SIGNED, 4, -8),
            (SignednessSemantics.SIGNED, 4, 0),
            (SignednessSemantics.SIGNLESS, 4, -8),
            (SignednessSemantics.SIGNLESS, 4, 15),
            (SignednessSemantics.SIGNLESS, 4, 0),
        ],
    )
    def test_validate_signed(self, signedness, bitwidth, value):
        """Test validation of a signed integer type."""
        int_type = IntegerType(bitwidth=bitwidth, signedness=signedness)
        int_type.validate(value)

    @pytest.mark.parametrize(
        "signedness, bitwidth, value",
        [
            (SignednessSemantics.SIGNED, 4, -9),
            (SignednessSemantics.SIGNED, 4, 8),
            (SignednessSemantics.UNSIGNED, 4, -1),
            (SignednessSemantics.UNSIGNED, 4, 16),
            (SignednessSemantics.SIGNLESS, 4, -9),
            (SignednessSemantics.SIGNLESS, 4, 16),
            (SignednessSemantics.SIGNLESS, 4, 1.23),
        ],
    )
    def test_validate_signed_fail(self, signedness, bitwidth, value):
        """Test validation failure of a signed integer type."""
        int_type = IntegerType(bitwidth=bitwidth, signedness=signedness)
        with pytest.raises(ValueError):
            int_type.validate(value)


class TestFloatType:
    def test_bitwidth(self):
        """Test getting a float type with a specific bitwidth."""
        float_type = FloatType(FloatTypeKind.F32)
        assert float_type.bitwidth == 32

    def test_str(self):
        """Test the string representation of a float type."""
        float_type = FloatType(FloatTypeKind.F64)
        assert str(float_type) == "f64"

    def test_validate(self):
        """Test validation of a float type."""
        float_type = FloatType(FloatTypeKind.F32)
        float_type.validate(3.14)
        float_type.validate(-2.71)

    @pytest.mark.parametrize("value", ["not a float", 3, None])
    def test_validate_fail(self, value):
        """Test validation failure of a float type."""
        float_type = FloatType(FloatTypeKind.F32)
        with pytest.raises(ValueError):
            float_type.validate(value)
