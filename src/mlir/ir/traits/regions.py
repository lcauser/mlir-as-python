from pydantic import BaseModel, field_validator
from .base import OpTrait
from functools import lru_cache


@lru_cache(maxsize=None)
def NRegions(n: int):
    """Factory for producing classes that enforce a fixed number of regions."""

    class _NRegions(BaseModel, OpTrait):
        @field_validator("regions")
        @classmethod
        def check_regions_have_correct_length(cls, v):
            if len(v) != n:
                raise ValueError(
                    f"{cls.__name__} requires exactly {n} regions, but got {len(v)}"
                )
            return v

    return _NRegions


ZeroRegions = NRegions(0)
OneRegion = NRegions(1)
TwoRegions = NRegions(2)


class VariadicRegions(OpTrait):
    """Trait for operations with a variadic number of regions."""

    ...
