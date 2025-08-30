from .base import OpTrait
from pydantic import BaseModel, Field, field_validator


class Symbol(OpTrait, BaseModel):
    """Trait that marks operations as defining symbols."""

    @field_validator("attributes")
    @classmethod
    def check_has_symbol_attribute(cls, v):
        if "sym_name" not in v:
            raise ValueError(f"{cls.__name__} requires a 'sym_name' attribute.")
        return v


class SymbolTable(OpTrait, BaseModel):
    """Trait that marks operations as having and maintaining a symbol table."""

    symbol_table: dict[str, Symbol] = Field(default_factory=dict)
    """Dictionary mapping symbol names to their defining operations."""
