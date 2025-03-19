# The Software is copyright (c) Commonwealth Scientific and Industrial Research Organisation (CSIRO) 2023-2025.
from collections.abc import Generator
from typing import Optional

import pytest

from dief_competition_2.classify import ExampleClass
from dief_competition_2.init_utils.config import get_logger

logger = get_logger()


class TestExample:
    """
    Example unit test class, following pytest conventions (https://docs.pytest.org/en/7.1.x/explanation/goodpractices.html#test-discovery).

    To make pytest actually find and run your tests you need to:
    1. prefix your class name with `Test` (E.g. `class TestExample`)
    2. prefix all your test functions with `test_` (eg `def test_example_function()`)

    To quote the pytest doc: `pytest discovers all tests following its [Conventions for Python test discovery]
    (https://docs.pytest.org/en/7.1.x/explanation/goodpractices.html#test-discovery)),
    so it finds both test_ prefixed functions. There is no need to subclass anything, but make sure to prefix
    your class with Test otherwise the class will be skipped.`
    """

    test_place: Optional[str] = None

    @pytest.fixture(scope="function")
    def example_class(self) -> Generator[ExampleClass, None, None]:
        """Fixture to create an ExampleClass instance for each test function.
        The instance yielded by this fixture is supplied to each test function with a parameter matching this function name.

        The fixture's 'scope' is set to 'function', so it is created and destroyed for each test function.
        Other options are class, module, package or session

        See https://docs.pytest.org/en/7.1.x/how-to/fixtures.html for more detail.
        """
        # initialise stuff here, and yield it to any test function that requests it
        ex = ExampleClass("a Unit Test")
        yield ex

    def test_example_returns_string(self, example_class: ExampleClass) -> None:
        """Tests that the function returns a string
        The example_class fixture is created by the fixture above, and passed to this function as a parameter.
        """
        logger.info(example_class)
        result = example_class.print_hi_from_place("A Test")
        assert isinstance(result, str), "print_hi_from_place() should only return a string"

    def test_example_instantiation(self) -> None:
        """Just tests that Example can be created"""
        ex = ExampleClass("Created!")
        assert isinstance(ex, ExampleClass)

    def test_example_result_correct(self, example_class: ExampleClass) -> None:
        """Tests that the function returns a string
        This function gets a _new_ instance of the example_class fixture, so it is not the same as the one in the previous test.
        """
        input = "Another Test"
        result = example_class.print_hi_from_place(input)
        assert input in result, "Out input string should be contained in the result"
