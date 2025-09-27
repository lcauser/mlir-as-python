import pytest

from mlir.utils.validator import ValidatorError, ValidatorMeta, validator


class TestValidator:
    def test_validator_adds_attribute(self):
        def my_func():
            return None

        my_func = validator(my_func)
        # slight abuse of accessing private attributes here but ay-oh
        assert hasattr(my_func, "_is_validator")
        assert my_func._is_validator is True

    class MyClass(metaclass=ValidatorMeta):
        def __init__(self, x, lower_bound, upper_bound):
            self.x = x
            self.lower_bound = lower_bound
            self.upper_bound = upper_bound

        @validator
        def check_lower_bound(self):
            if self.x < self.lower_bound:
                raise ValueError("x must be greater than or equal to lower_bound")

        @validator
        def check_upper_bound(self):
            if self.x > self.upper_bound:
                raise ValueError("x must be less than or equal to upper_bound")

        @validator
        def check_real(self):
            if not isinstance(self.x, (int, float)):
                raise ValueError("x must be a real number")

        @validator
        def check_upper_bound_is_real(self):
            if not isinstance(self.upper_bound, (int, float)):
                raise ValueError("upper_bound must be a real number")

    def test_validator_raises_value_error(self):
        with pytest.raises(
            ValidatorError, match="x must be greater than or equal to lower_bound"
        ):
            self.MyClass(-2, 0, 10)

    def test_validator_raises_multiple_errors(self):
        with pytest.raises(ValidatorError) as exc_info:
            self.MyClass(-5, 0, 10 + 1j)

        errors = exc_info.value.errors
        assert len(errors) >= 2
        assert any(
            "x must be greater than or equal to lower_bound" in str(e) for e in errors
        )
        assert any("upper_bound must be a real number" in str(e) for e in errors)

    def test_validator_is_successful(self):
        obj = self.MyClass(10, 0, 20)
        assert obj.x == 10
