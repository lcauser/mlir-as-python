from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ir.types import TypeBase


class MLIRContext:
    """Represents the compilation context for MLIR.

    Features will not be anticipated, but added as needed.

    This handles:

    * Types used: Declares one instance of each type to allow for effecient memory usage,
      but effecient lowering etc.
    """

    def __init__(self):
        """Instantiate a new MLIRContext."""
        self._types: dict[tuple, "TypeBase"] = {}

    def get_type(self, key: tuple) -> "TypeBase | None":
        """Return a type from the context by a key, which is a tuple of its type and
        parameters that define the type. If the type does not exist, return None."""
        return self._types.get(key, None)

    def add_type(self, key: tuple, value: "TypeBase"):
        """Add a type to the context. If the type already exists, this will raise an
        error."""
        if key in self._types:
            if self._types[key] == value:
                raise ValueError(f"Type {value} already exists in the context.")
            else:
                raise ValueError(
                    f"The {key} already exists in the context with a different value."
                )
        self._types[key] = value
