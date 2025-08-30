from pydantic import BaseModel, field_validator
from functools import lru_cache
from .base import OpTrait


@lru_cache(maxsize=None)
def NResults(n: int):
    """Factory for producing classes that enforce a fixed number of results."""

    class _NResults(BaseModel, OpTrait):
        @field_validator("results")
        @classmethod
        def check_results_have_correct_length(cls, v):
            if len(v) != n:
                raise ValueError(
                    f"{cls.__name__} requires exactly {n} operands, but got {len(v)}"
                )
            return v

    return _NResults


ZeroResults = NResults(0)
OneResult = NResults(1)
TwoResults = NResults(2)


class VariadicResults(OpTrait):
    """Trait for operations with a variadic number of results."""

    ...
