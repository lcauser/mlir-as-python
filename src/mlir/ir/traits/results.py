from mlir.utils.validator import validator
from functools import lru_cache
from .base import OpTrait
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlir.ir.operations import Operation


@lru_cache(maxsize=None)
def NResults(n: int):
    """Factory for producing classes that enforce a fixed number of results."""

    class _NResults(OpTrait):
        """Trait for operations with exactly `n` results."""

        @validator
        def validate_results_have_correct_length(self: "Operation") -> "Operation":
            if len(self.results) != n:
                raise ValueError(
                    f"{self.__class__.__name__} requires exactly {n} results, but got "
                    f"{len(self.results)}"
                )
            return self

    return _NResults


ZeroResults = NResults(0)
OneResult = NResults(1)
TwoResults = NResults(2)


class VariadicResults(OpTrait):
    """Trait for operations with a variadic number of results."""

    # You can add additional validators or documentation here if
