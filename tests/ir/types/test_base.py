import pytest
from mlir.context import MLIRContext
from mlir.ir.types import TypeBase
from inspect import signature
from itertools import product

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

        test_parameters = type_class.__test_parameters__
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
