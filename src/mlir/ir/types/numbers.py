from pydantic import Field, PositiveInt
from functools import cached_property
from enum import Enum

from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
        """Note: __init__ is specified so the signature can be used for factory testing."""
        super().__init__(bitwidth=bitwidth, signedness=signedness)

    @classmethod
    def get_signed(cls, context: "MLIRContext", bitwidth: PositiveInt) -> "IntegerType":
        """Return a signed integer type with the specified bit width."""
        return cls.get(context, bitwidth, SignednessSemantics.SIGNED)

    @classmethod
    def get_unsigned(
        cls, context: "MLIRContext", bitwidth: PositiveInt
    ) -> "IntegerType":
        """Return an unsigned integer type with the specified bit width."""
        return cls.get(context, bitwidth, SignednessSemantics.UNSIGNED)

    def validate(self, value):
        """Validates that the value is an integer within the bounds of the type.

        * For signed integers, the range is [-2^(bitwidth-1), 2^(bitwidth-1) - 1].
        * For unsigned integers, the range is [0, 2^bitwidth - 1].
        * For signless integers, the bounds take the minimum of both both lower bounds and
          the maximum of both upper bounds, with type interpretted at compile time.
          Practically, we must check for [-2^(bitwidth-1), 2^bitwidth - 1].
        """
        if not isinstance(value, int):
            raise ValueError(f"Value {value} is not an integer.")

        lower_bound = (
            0
            if self.signedness == SignednessSemantics.UNSIGNED
            else -(2 ** (self.bitwidth - 1))
        )
        upper_bound = (
            2 ** (self.bitwidth - 1) - 1
            if self.signedness == SignednessSemantics.SIGNED
            else 2**self.bitwidth - 1
        )
        if not (lower_bound <= value <= upper_bound):
            raise ValueError(
                f"Value {value} is out of bounds for {self.signedness.value} i{self.bitwidth}."
            )

    def __str__(self) -> str:
        return f"i{self.bitwidth}"

    def __hash__(self) -> int:
        return hash((self.__class__, self.bitwidth, self.signedness))

    __test_parameters__ = dict(
        bitwidth=[1, 2, 7, 128], signedness=[member for member in SignednessSemantics]
    )
    __test_validate_passes__ = [
        dict(bitwidth=bitwidth, signedness=signedness, value=value)
        for signedness, bitwidth, value in [
            (SignednessSemantics.UNSIGNED, 4, 0),
            (SignednessSemantics.UNSIGNED, 4, 15),
            (SignednessSemantics.SIGNED, 4, 7),
            (SignednessSemantics.SIGNED, 4, -8),
            (SignednessSemantics.SIGNED, 4, 0),
            (SignednessSemantics.SIGNLESS, 4, -8),
            (SignednessSemantics.SIGNLESS, 4, 15),
            (SignednessSemantics.SIGNLESS, 4, 0),
        ]
    ]
    __test_validate_fails__ = [
        dict(bitwidth=bitwidth, signedness=signedness, value=value)
        for signedness, bitwidth, value in [
            (SignednessSemantics.SIGNED, 4, -9),
            (SignednessSemantics.SIGNED, 4, 8),
            (SignednessSemantics.UNSIGNED, 4, -1),
            (SignednessSemantics.UNSIGNED, 4, 16),
            (SignednessSemantics.SIGNLESS, 4, -9),
            (SignednessSemantics.SIGNLESS, 4, 16),
            (SignednessSemantics.SIGNLESS, 4, 1.23),
        ]
    ]


class IndexType(TypeBase):
    """Represents an index type in MLIR, used in loop bounds, indexing and dimensions."""

    def __init__(self):
        """Note: __init__ is specified so the signature can be used for factory testing."""
        super().__init__()

    def validate(self, value):
        """Not sure this is entirely correct."""
        if not isinstance(value, int):
            raise ValueError(f"Value {value} is not an integer.")
        if value < 0:
            raise ValueError(f"Value {value} cannot be negative for index type.")

    def __str__(self) -> str:
        return "index"

    def __hash__(self) -> int:
        return hash(self.__class__)


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
        """Note: __init__ is specified so the signature can be used for factory testing."""
        super().__init__(kind=kind)

    @cached_property
    def bitwidth(self) -> int:
        """Return the bit width of the floating-point type."""
        return _float_bitwidths[self.kind]

    def validate(self, value):
        """Validates that the value is a float.

        More details to be added later."""
        if not isinstance(value, float):
            raise ValueError(f"Value {value} is not a float.")

    def __str__(self) -> str:
        """Return a string representation of the floating-point type."""
        return f"{self.kind.value}"

    def __hash__(self) -> int:
        """Hash the float type based on its kind."""
        return hash((self.__class__, self.kind))

    __test_parameters__ = dict(kind=[member for member in FloatTypeKind])
    __test_validate_passes__ = [dict(kind=kind, value=3.14) for kind in FloatTypeKind]
    __test_validate_fails__ = [
        dict(kind=kind, value="not a float") for kind in FloatTypeKind
    ]
