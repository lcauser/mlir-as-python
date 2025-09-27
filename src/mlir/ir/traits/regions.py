from functools import lru_cache
from typing import TYPE_CHECKING

from mlir.utils.validator import validator

from .base import OpTrait

if TYPE_CHECKING:
    from mlir.ir.operations import Operation


@lru_cache(maxsize=None)
def NRegions(n: int):
    """Factory for producing classes that enforce a fixed number of regions."""

    class _NRegions(OpTrait):
        """Trait for operations with exactly `n` results."""

        @validator
        def validate_regions_have_correct_length(self: "Operation") -> "Operation":
            if len(self.regions) != n:
                raise ValueError(
                    f"{self.__class__.__name__} requires exactly {n} regions, but got "
                    f"{len(self.regions)}"
                )
            return self

    return _NRegions


ZeroRegions = NRegions(0)
OneRegion = NRegions(1)
TwoRegions = NRegions(2)
