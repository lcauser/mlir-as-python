from mlir.ir.attributes import AttributeBase, AttributeStorage
from mlir.ir.types import TypeBase, TypeStorage


class MLIRContext:
    """Represents the compilation context for MLIR.

    Features will not be anticipated, but added as needed.

    This handles:

    * Types used: Declares one instance of each type to allow for effecient memory usage,
      but effecient lowering etc.
    * Attributes used: Same as types, but for attributes.
    """

    def __init__(self):
        """Instantiate a new MLIRContext."""
        self.types: TypeStorage = TypeStorage()
        self.attributes: AttributeStorage = AttributeStorage()

    def get_type(self, key: tuple) -> TypeBase | None:
        """Return a type from the context by a key, which is a tuple of its type and
        parameters that define the type. If the type does not exist, return None."""
        return self.types.get(key)

    def add_type(self, key: tuple, value: TypeBase):
        """Add a type to the context. If the type already exists, this will raise an
        error."""
        if key in self.types:
            if self.types[key] == value:
                raise ValueError(f"Type {value} already exists in the context.")
            else:
                raise ValueError(
                    f"The {key} already exists in the context with a different value."
                )
        self.types.add(key, value)

    def get_attribute(self, key: tuple) -> AttributeBase | None:
        """Return an attribute from the context by a key, which is a tuple of its type and
        parameters that define the type. If the attribute does not exist, return None."""
        return self.attributes.get(key)

    def add_attribute(self, key: tuple, value: AttributeBase):
        """Add an attribute to the context. If the attribute already exists, this will raise
        an error."""
        self.attributes.add(key, value)
