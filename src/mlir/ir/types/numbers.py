from pydantic import Field, PositiveInt
from functools import cached_property
from enum import Enum

from mlir.context import MLIRContext

from .base import TypeBase


class SignednessSemantics(Enum):
    """Enumeration for signedness semantics in MLIR."""

    SIGNLESS = "signless"
    """Represents a signless integer type, which does not specify signedness."""

    UNSIGNED = "unsigned"
    """Represents an unsigned integer type, which can only represent non-negative values."""

    SIGNED = "signed"
    """Represents a signed integer type, which can represent both negative and non-negative
    values. The method of implementation is not specified, and left to the target."""


class IntegerType(TypeBase):
    """Represents an integer type in MLIR."""

    bitwidth: PositiveInt = Field(..., description="The bit width of the integer type.")
    signedness: SignednessSemantics = Field(
        ..., description="The signedness semantics of the integer type."
    )

    def __init__(
        self,
        bitwidth: PositiveInt,
        signedness: SignednessSemantics = SignednessSemantics.SIGNLESS,
    ):
        super().__init__(bitwidth=bitwidth, signedness=signedness)

    @classmethod
    def get_signed(cls, context: MLIRContext, bitwidth: PositiveInt) -> "IntegerType":
        """Return a signed integer type with the specified bit width."""
        return cls.get(context, bitwidth, SignednessSemantics.SIGNED)

    @classmethod
    def get_unsigned(cls, context: MLIRContext, bitwidth: PositiveInt) -> "IntegerType":
        """Return an unsigned integer type with the specified bit width."""
        return cls.get(context, bitwidth, SignednessSemantics.UNSIGNED)

    __test_parameters__ = dict(
        bitwidth=[1, 2, 7, 128], signedness=[member for member in SignednessSemantics]
    )

    def __str__(self) -> str:
        return f"i{self.bitwidth}"


class IndexType(TypeBase):
    """Represents an index type in MLIR, used in loop bounds, indexing and dimensions."""

    def __init__(self):
        super().__init__()


class FloatTypeKind(Enum):
    """Different types of floats are defined explicitly."""

    F16 = "f16"
    """16-bit floating-point type."""

    F32 = "f32"
    """32-bit floating-point type."""

    F64 = "f64"
    """64-bit floating-point type."""

    F80 = "f80"
    """80-bit floating-point type, often used in x86 floating-point operations."""

    F128 = "f128"
    """128-bit floating-point type, often used in high-precision calculations."""

    BF16 = "bf16"
    """A special 16-bit type of float frequently used to accelerate machine learning."""


_float_bitwidths = {
    FloatTypeKind.F16: 16,
    FloatTypeKind.F32: 32,
    FloatTypeKind.F64: 64,
    FloatTypeKind.F80: 80,
    FloatTypeKind.F128: 128,
    FloatTypeKind.BF16: 16,
}


class FloatType(TypeBase):
    """Represents a floating-point type in MLIR."""

    kind: FloatTypeKind = Field(..., description="The kind of floating-point type.")

    def __init__(self, kind: FloatTypeKind):
        super().__init__(kind=kind)

    @cached_property
    def bitwidth(self) -> int:
        """Return the bit width of the floating-point type."""
        return _float_bitwidths[self.kind]

    def __str__(self) -> str:
        """Return a string representation of the floating-point type."""
        return f"{self.kind.value}"

    __test_parameters__ = dict(kind=[member for member in FloatTypeKind])
