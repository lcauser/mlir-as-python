from functools import lru_cache
from typing import TYPE_CHECKING

from mlir.utils.validator import validator

from .base import OpTrait

if TYPE_CHECKING:
    from mlir.ir.operations import Operation


@lru_cache(maxsize=None)
def NOperands(n: int):
    """Factory for producing classes that enforce a fixed number of operands."""

    class _NOperands(OpTrait):
        """Trait for operations with exactly `n` operands."""

        @validator
        def validate_operands_have_correct_length(self: "Operation") -> "Operation":
            if len(self.operands) != n:
                raise ValueError(
                    f"{self.__class__.__name__} requires exactly {n} operands, but got "
                    f"{len(self.operands)}"
                )
            return self

    return _NOperands


ZeroOperands = NOperands(0)
OneOperand = NOperands(1)
TwoOperands = NOperands(2)


class VariadicOperands(OpTrait):
    """Trait for operations with a variadic number of operands."""

    ...
