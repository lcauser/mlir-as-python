from pydantic import BaseModel, ConfigDict, Field, model_validator
from ir.types import TypeBase


class AttributeBase(BaseModel):
    """Base class for MLIR attributes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    attribute_type: TypeBase = Field(..., alias="type")
    """The type of the attribute, which is an instance of TypeBase."""

    value: any
    """The value of the attribute: subclasses should use stricter type checking."""

    @model_validator(mode="before")
    def validate_type(cls, values):
        """Ensure that the type is an instance of TypeBase."""
        values.get("attribute_type").validate(values.get("value"))


class AttributeStorage:
    """A storage for attributes for deduplication."""

    def __init__(self):
        self._attributes = {}

    def get(self, key: tuple) -> AttributeBase | None:
        """Return an attribute from the context by a key, which is a tuple of its type and
        parameters that define the type. If the attribute does not exist, return None."""
        return self._attributes.get(key, None)

    def add(self, key: tuple, value: AttributeBase):
        """Add an attribute to the context. If the attribute already exists, this will raise
        an error."""
        if key in self._attributes:
            if self._attributes[key] == value:
                raise ValueError(f"Attribute {value} already exists in the context.")
            else:
                raise ValueError(
                    f"The {key} already exists in the context with a different value."
                )
        self._attributes[key] = value

    def __contains__(self, key: tuple) -> bool:
        return key in self._attributes

    def __getitem__(self, key: tuple) -> AttributeBase:
        return self._attributes[key]
