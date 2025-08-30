from pydantic import BaseModel, field_validator
from functools import lru_cache
from .base import OpTrait


@lru_cache(maxsize=None)
def NOperands(n: int):
    """Factory for producing classes that enforce a fixed number of operands."""

    class _NOperands(BaseModel, OpTrait):
        @field_validator("operands")
        @classmethod
        def check_operands_have_correct_length(cls, v):
            if len(v) != n:
                raise ValueError(
                    f"{cls.__name__} requires exactly {n} operands, but got {len(v)}"
                )
            return v

    return _NOperands


ZeroOperands = NOperands(0)
OneOperand = NOperands(1)
TwoOperands = NOperands(2)


class VariadicOperands(OpTrait):
    """Trait for operations with a variadic number of operands."""

    ...
