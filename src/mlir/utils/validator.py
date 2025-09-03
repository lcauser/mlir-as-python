def validator(method):
    """This decorates methods with a validator, which when combined with the ValidatorMeta
    metaclass, will run the method after __init__."""
    method._is_validator = True
    return method


class ValidatorError(Exception):
    """Aggregate multiple validation errors into one exception."""

    def __init__(self, errors: list[Exception]):
        self.errors = errors
        super().__init__(str(self))

    def __str__(self):
        messages = "\n".join(f"    {str(e)}" for e in self.errors)
        return f"Validation failed with the following errors:\n{messages}"


class ValidatorMeta(type):
    """Metaclass for adding the functionality to add validators to a class that run on
    instantiation.

    Validators can be added to a class by adding the @validator decorator to a method.
    They are expected to be able to execute in a commutative way: the order the validators
    run shouldn't matter. This allows us to flexibly add validators in subclasses, and mix
    in validation from traits.

    .. code-block:: python

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
    """

    def __new__(cls, name, bases, namespace):
        validators = []
        for attr_value in namespace.values():
            if getattr(attr_value, "_is_validator", False):
                validators.append(attr_value)
        namespace["_validators"] = validators

        def __post_init__(self, *args, **kwargs):
            errors = []
            for validator in self._validators:
                try:
                    validator(self)
                except Exception as e:
                    errors.append(e)

            if errors:
                raise ValidatorError(errors)

        namespace["__post_init__"] = __post_init__

        new_cls = super().__new__(cls, name, bases, namespace)
        orig_init = getattr(new_cls, "__init__", None)

        def __init__(self, *args, **kwargs):
            if orig_init is not object.__init__:
                orig_init(self, *args, **kwargs)
            # Only call __post_init__ if this is the most derived class
            if type(self) is new_cls:
                self.__post_init__(*args, **kwargs)

        setattr(new_cls, "__init__", __init__)
        return new_cls
