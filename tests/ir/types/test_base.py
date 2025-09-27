from copy import deepcopy
from inspect import signature
from itertools import product

import pytest

from mlir.context import MLIRContext
from mlir.ir.types import TypeBase

all_types = TypeBase.__subclasses__()


@pytest.mark.parametrize("type_class", all_types, scope="class")
class TestTypes:
    """Uses a factory pattern to systematically create and test types. Valid parameters
    are provided by the subclasses to allow for easy testing."""

    @pytest.fixture(scope="class", autouse=True)
    def combinations(self, type_class):
        """Get all combinations of parameters for a type class."""
        args = [p for p in signature(type_class.__init__).parameters if p != "self"]
        if len(args) == 0:
            return []

        if (
            test_parameters := getattr(type_class, "__test_parameters__", None)
        ) is None:
            return []
        assert len(test_parameters) == len(args)
        keys = test_parameters.keys()
        values = test_parameters.values()
        return [dict(zip(keys, combination)) for combination in product(*values)]

    def test_instantiation(self, type_class, combinations):
        """Test instantiation of all types."""

        for params in combinations:
            type_instance = type_class(**params)
            assert isinstance(type_instance, type_class)
            for key, value in params.items():
                assert getattr(type_instance, key) == value

    def test_get(self, type_class, combinations):
        """Test the get method of all types."""

        context = MLIRContext()
        for params in combinations:
            type_instance = type_class.get(context, *params.values())
            assert isinstance(type_instance, type_class)
            for key, value in params.items():
                assert getattr(type_instance, key) == value

            # Ensure the type is stored in the context
            retrieved_type = context.get_type(
                (type_class.__name__,) + tuple(params.values())
            )
            assert retrieved_type is type_instance

    def test_repr(self, type_class, combinations):
        """Test the __repr__ method of all types."""
        for params in combinations:
            type_instance = type_class(**params)
            repr_str = repr(type_instance)
            assert isinstance(repr_str, str)
            assert type_instance.__class__.__name__ in repr_str
            for key, value in params.items():
                assert f"{key}={value}" in repr_str

    def test_validate_passes(self, type_class):
        """Test the validate method of all types with valid values."""
        if not hasattr(type_class, "__test_validate_passes__"):
            return
        for test in type_class.__test_validate_passes__:
            test = deepcopy(test)
            value = test.pop("value")
            type_instance = type_class(**test)
            type_instance.validate_type(value)

    def test_validate_fails(self, type_class):
        """Test the validate method of all types with invalid values."""
        if not hasattr(type_class, "__test_validate_fails__"):
            return
        for test in type_class.__test_validate_fails__:
            test = deepcopy(test)
            value = test.pop("value")
            type_instance = type_class(**test)
            with pytest.raises(ValueError):
                type_instance.validate_type(value)
