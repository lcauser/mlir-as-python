from mlir.ir.attributes import AttributeBase
import pytest
from copy import deepcopy

all_types = AttributeBase.__subclasses__()


@pytest.mark.parametrize("attribute_class", all_types, scope="class")
class TestAttributeBase:
    """Uses a factory pattern to systematically create and test attributes. Valid parameters
    are provided by the subclasses to allow for easy testing."""

    def test_validation_passes(self, attribute_class):
        type_class = attribute_class.model_fields["attribute_type"].annotation
        if not hasattr(type_class, "__test_validate_passes__"):
            return
        for test in type_class.__test_validate_passes__:
            test = deepcopy(test)
            value = test.pop("value")
            type_instance = type_class(**test)
            attribute_class(type=type_instance, value=value)

    def test_validation_fails(self, attribute_class):
        type_class = attribute_class.model_fields["attribute_type"].annotation
        if not hasattr(type_class, "__test_validate_fails__"):
            return
        for test in type_class.__test_validate_fails__:
            test = deepcopy(test)
            value = test.pop("value")
            type_instance = type_class(**test)
            with pytest.raises(ValueError):
                attribute_class(type=type_instance, value=value)
