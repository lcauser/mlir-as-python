from pydantic import Field
from mlir.ir.types import IntegerType, IndexType, FloatType
from .base import AttributeBase


class IntegerAttribute(AttributeBase):
    """An attribute representing an integer value in MLIR."""

    attribute_type: IntegerType = Field(..., alias="type")
    """The type of the integer attribute, which is an instance of IntegerType."""

    value: int
    """The integer value of the attribute."""

    def __str__(self) -> str:
        return f"{self.value} : {self.attribute_type}"


class IndexAttribute(AttributeBase):
    """An attribute representing an index value in MLIR."""

    attribute_type: IndexType = Field(..., alias="type")
    """The type of the index attribute, which is an instance of IndexType."""

    value: int
    """The index value of the attribute."""

    def __str__(self) -> str:
        return f"{self.value} : {self.attribute_type}"


class FloatAttribute(AttributeBase):
    """An attribute representing a float value in MLIR."""

    attribute_type: FloatType = Field(..., alias="type")
    """The type of the float attribute, which is an instance of FloatType."""

    value: float
    """The float value of the attribute."""

    def __str__(self) -> str:
        return f"{self.value} : {self.attribute_type}"
