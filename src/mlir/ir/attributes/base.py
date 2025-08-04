from pydantic import BaseModel, ConfigDict


class AttributeBase(BaseModel):
    """Base class for MLIR attributes."""

    model_config = ConfigDict(extra="forbid", frozen=True)
